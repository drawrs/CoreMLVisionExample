//
//  ViewController.swift
//  CoreMLExample
//
//  Created by Rizal Hilman on 19/05/20.
//  Copyright © 2020 Rizal Hilman. All rights reserved.
//

import UIKit
import AVFoundation
import Vision

class ViewController: UIViewController {

    @IBOutlet weak var viewCamera: UIView!
    @IBOutlet weak var labelResult: UILabel!
    @IBOutlet weak var switchVision: UISwitch!
    
    private let coreMLModel = Inceptionv3()
    private var requests = [VNRequest]()
    
    private let session = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer! = nil
    private let videoDataOutput = AVCaptureVideoDataOutput()
    
    private let videoDataOutputQueue = DispatchQueue(label: "VideoDataOutput", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem)
    
    /// only support back camera
    var exifOrientationFromDeviceOrientation: Int32 {
        let exifOrientation: DeviceOrientation
        enum DeviceOrientation: Int32 {
            case top0ColLeft = 1
            case top0ColRight = 2
            case bottom0ColRight = 3
            case bottom0ColLeft = 4
            case left0ColTop = 5
            case right0ColTop = 6
            case right0ColBottom = 7
            case left0ColBottom = 8
        }
        switch UIDevice.current.orientation {
        case .portraitUpsideDown:
            exifOrientation = .left0ColBottom
        case .landscapeLeft:
            exifOrientation = .top0ColLeft
        case .landscapeRight:
            exifOrientation = .bottom0ColRight
        default:
            exifOrientation = .right0ColTop
        }
        return exifOrientation.rawValue
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setuView()
        setupAVCapture()
        setupVision()
        
        // start the capture
        startCaptureSession()
    }
    
    func setuView(){
        labelResult.layer.cornerRadius = 10
        labelResult.layer.masksToBounds = true
    }
    
    // MARK: Setup The Camera View
    func setupAVCapture() {
        var deviceInput: AVCaptureDeviceInput!
        // Select a video device, make an input
        let videoDevice = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .back).devices.first
        do {
            deviceInput = try AVCaptureDeviceInput(device: videoDevice!)
        } catch {
            print("Could not create video device input: \(error)")
            return
        }
        
        session.beginConfiguration()
        session.sessionPreset = .vga640x480 // Model image size is smaller.
        
        // Add a video input
        guard session.canAddInput(deviceInput) else {
            print("Could not add video device input to the session")
            session.commitConfiguration()
            return
        }
        
        session.addInput(deviceInput)
        if session.canAddOutput(videoDataOutput) {
            session.addOutput(videoDataOutput)
            // Add a video data output
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as AnyHashable as! String: NSNumber(value: kCVPixelFormatType_32BGRA)]
            videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
        } else {
            print("Could not add video data output to the session")
            session.commitConfiguration()
            return
        }
        
        session.commitConfiguration()
        
        // MARK: Add camera preview to uiview
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
        previewLayer.frame = viewCamera.layer.bounds
        viewCamera.layer.addSublayer(previewLayer)
        
    }
    // MARK: For starting the camera
    func startCaptureSession() {
        session.startRunning()
    }
    
    // MARK: Clean up capture setup
    func teardownAVCapture() {
        previewLayer.removeFromSuperlayer()
        previewLayer = nil
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(true)
        teardownAVCapture()
    }
    
    // MARK: Setup Vision Framework
    func setupVision() {
        guard let visionModel = try? VNCoreMLModel(for: inceptionv3model.model) else {
            fatalError("can't load Vision ML model")
        }
        // MARK: Vision CoreML Prediction
        let classificationRequest = VNCoreMLRequest(model: visionModel) { (request: VNRequest, error: Error?) in
            guard let observations = request.results else {
                print("no results:\(error!)")
                return
            }
            let classifications = observations[0...1] // number of model labels (0...1 means 0 - 1 = 2 labels)
                .compactMap({ $0 as? VNClassificationObservation })
                .filter({ $0.confidence > 0.4 })
                .map({ "\($0.identifier) \($0.confidence)" })
            
            // MARK: Set prediction result to the label
            DispatchQueue.main.async {
                self.labelResult.text = classifications.joined(separator: "\n")
            }
        }
        classificationRequest.imageCropAndScaleOption = VNImageCropAndScaleOption.centerCrop
        
        self.requests = [classificationRequest]
    }
}

extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    // MARK: Handling video output (every frame)
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Handle the video output
        if connection.videoOrientation != .portrait {
            connection.videoOrientation = .portrait
            return
        }
        
        DispatchQueue.main.async {
            if self.switchVision.isOn {
                // Use Vision
                self.handleImageBufferWithVision(imageBuffer: sampleBuffer)
            }
            else {
                // Use Core ML
                self.handleImageBufferWithCoreML(imageBuffer: sampleBuffer)
            }
        }
    }
    
    // MARK: Handling With Vision Framework
    func handleImageBufferWithVision(imageBuffer: CMSampleBuffer) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(imageBuffer) else {
            return
        }
        
        var requestOptions:[VNImageOption : Any] = [:]
        
        if let cameraIntrinsicData = CMGetAttachment(imageBuffer, key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, attachmentModeOut: nil) {
            requestOptions = [.cameraIntrinsics:cameraIntrinsicData]
        }
        
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: CGImagePropertyOrientation(rawValue: UInt32(self.exifOrientationFromDeviceOrientation))!, options: requestOptions)
        do {
            try imageRequestHandler.perform(self.requests)
        } catch {
            print(error)
        }
    }
    
    // MARK: Handling Image With CoreML
    func handleImageBufferWithCoreML(imageBuffer: CMSampleBuffer) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(imageBuffer) else {
            return
        }
        do {
            let prediction = try self.inceptionv3model.prediction(image: self.resize(pixelBuffer: pixelBuffer)!)
            
            DispatchQueue.main.async {
                if let prob = prediction.classLabelProbs[prediction.classLabel] {
                    self.labelResult.text = "\(prediction.classLabel) \(String(describing: prob))"
                }
            }
        }
        catch let error as NSError {
            fatalError("Unexpected error ocurred: \(error.localizedDescription).")
        }
    }
    
    /// resize CVPixelBuffer
    /// - Parameter pixelBuffer: CVPixelBuffer by camera output
    /// - Returns: CVPixelBuffer with size (299, 299)
    func resize(pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        let imageSide = 299
        var ciImage = CIImage(cvPixelBuffer: pixelBuffer, options: nil)
        let transform = CGAffineTransform(scaleX: CGFloat(imageSide) / CGFloat(CVPixelBufferGetWidth(pixelBuffer)), y: CGFloat(imageSide) / CGFloat(CVPixelBufferGetHeight(pixelBuffer)))
        ciImage = ciImage.transformed(by: transform).cropped(to: CGRect(x: 0, y: 0, width: imageSide, height: imageSide))
        let ciContext = CIContext()
        var resizeBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, imageSide, imageSide, CVPixelBufferGetPixelFormatType(pixelBuffer), nil, &resizeBuffer)
        ciContext.render(ciImage, to: resizeBuffer!)
        return resizeBuffer
    }
}
