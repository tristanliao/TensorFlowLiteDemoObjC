//
//  ViewController.m
//  TensorFlowLiteDemoObjC
//
//  Created by Bang Chiang Liao on 2019/8/7.
//  Copyright © 2019 Bang Chiang Liao. All rights reserved.
//

#import "ViewController.h"
#import <AVFoundation/AVFoundation.h>
#import <TFLTensorFlowLite/TFLTensorFlowLite.h>

@interface ViewController () <AVCaptureVideoDataOutputSampleBufferDelegate>

@property (strong, nonatomic) AVCaptureSession *session;
@property (strong, nonatomic) AVCaptureDevice *inputDevice;
@property (strong, nonatomic) AVCaptureDeviceInput *deviceInput;
@property (strong, nonatomic) AVCaptureVideoPreviewLayer *previewLayer;

@property (weak, nonatomic) IBOutlet UIImageView *imageView;
@property (weak, nonatomic) IBOutlet UILabel *messageLabel;
@property (weak, nonatomic) IBOutlet UILabel *fpsLabel;

@property (strong, nonatomic) TFLInterpreter *interpreter;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    [self setupInterpreter];
    [self setupCamera];
}

- (void)setupInterpreter {
    NSError *error;
    NSString *path = [[NSBundle mainBundle] pathForResource:@"mobilenet_quant_v1_224" ofType:@"tflite"];
//    NSString *path = [[NSBundle mainBundle] pathForResource:@"PornDetectModel" ofType:@"tflite"];
    self.interpreter = [[TFLInterpreter alloc] initWithModelPath:path error:&error];
    
    if (![self.interpreter allocateTensorsWithError:&error]) {
        NSLog(@"Create interpreter error: %@", error);
    }
}

- (void)setupCamera {
    self.session = [[AVCaptureSession alloc] init];
    [self.session setSessionPreset:AVCaptureSessionPresetPhoto];
    
    self.inputDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    NSError *error;
    self.deviceInput = [AVCaptureDeviceInput deviceInputWithDevice:self.inputDevice error:&error];
    
    if ([self.session canAddInput:self.deviceInput]) {
        [self.session addInput:self.deviceInput];
    }
    
    self.previewLayer = [[AVCaptureVideoPreviewLayer alloc] initWithSession:self.session];
    [self.previewLayer setVideoGravity:AVLayerVideoGravityResizeAspectFill];
    CALayer *rootLayer = [[self view] layer];
    [rootLayer setMasksToBounds:YES];
    CGRect frame = self.view.frame;
    [self.previewLayer setFrame:frame];
    [rootLayer insertSublayer:self.previewLayer atIndex:0];
    
    AVCaptureVideoDataOutput *videoDataOutput = [AVCaptureVideoDataOutput new];
    
    NSDictionary *rgbOutputSettings = [NSDictionary
                                       dictionaryWithObject:[NSNumber numberWithInt:kCMPixelFormat_32BGRA]
                                       forKey:(id)kCVPixelBufferPixelFormatTypeKey];
    [videoDataOutput setVideoSettings:rgbOutputSettings];
    [videoDataOutput setAlwaysDiscardsLateVideoFrames:YES];
    dispatch_queue_t videoDataOutputQueue = dispatch_queue_create("VideoDataOutputQueue", DISPATCH_QUEUE_SERIAL);
    [videoDataOutput setSampleBufferDelegate:self queue:videoDataOutputQueue];
    
    if ([self.session canAddOutput:videoDataOutput])
        [self.session addOutput:videoDataOutput];
    [[videoDataOutput connectionWithMediaType:AVMediaTypeVideo] setEnabled:YES];
    
    [self.session startRunning];
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection {
    CVImageBufferRef cvImage = CMSampleBufferGetImageBuffer(sampleBuffer);
    CIImage *ciImage = [[CIImage alloc] initWithCVPixelBuffer:cvImage];
    
    
    size_t height = CVPixelBufferGetHeight(cvImage);
    size_t width = CVPixelBufferGetWidth(cvImage);
    
    CGRect videoRect = CGRectMake(0, 0, width, height);
    CGSize scaledSize = CGSizeMake(224, 224);
    
    // Create a rectangle that meets the output size's aspect ratio, centered in the original video frame
    CGRect centerCroppingRect = AVMakeRectWithAspectRatioInsideRect(scaledSize, videoRect);
    
    CVPixelBufferRef croppedAndScaled = [self createCroppedPixelBufferRef:cvImage
                                                                 cropRect:centerCroppingRect
                                                                scaleSize:scaledSize
                                                                  context:nil];
    
//    NSData *imageData = [self dataFromPixelBufferRef:croppedAndScaled];
    
    
    NSError *error;
    TFLTensor *inputTensor = [self.interpreter inputTensorAtIndex:0 error:&error];
    NSData *imageData = [self rgbDataFromBuffer:croppedAndScaled
                                      byteCount:224 * 224 * 3
                               isModelQuantized:inputTensor.dataType == TFLTensorDataTypeUInt8];
    
    [inputTensor copyData:imageData error:&error];
    
    [self.interpreter invokeWithError:&error];
    if (error) {
        NSLog(@"Error: %@", error);
    }
    
    TFLTensor *outputTensor = [self.interpreter outputTensorAtIndex:0 error:&error];
    NSData *outputData = [outputTensor dataWithError:&error];
    
    
    if (error) {
        NSLog(@"Error: %@", error);
    }
}

- (NSData *)dataFromPixelBufferRef:(CVPixelBufferRef)pixelBufferRef {
    CVPixelBufferLockBaseAddress(pixelBufferRef, 0);
    void *buffer = CVPixelBufferGetBaseAddress(pixelBufferRef);
    size_t length = CVPixelBufferGetDataSize(pixelBufferRef);
    CVPixelBufferUnlockBaseAddress(pixelBufferRef, 0);
    
    NSData *data = [NSData dataWithBytes:buffer length:length];
    return data;
}

//- (CVPixelBufferRef)cropAndCenteredPixelBufferRef:(CVPixelBufferRef)pixelBufferRef width:(CGFloat)width height:(CGFloat)height {
//
//}

- (CVPixelBufferRef)createCroppedPixelBufferRef:(CVPixelBufferRef)pixelBuffer cropRect:(CGRect)cropRect scaleSize:(CGSize)scaleSize context:(CIContext *)context {
//    assertCropAndScaleValid(pixelBuffer, cropRect, scaleSize);
    
    CIImage *image = [CIImage imageWithCVImageBuffer:pixelBuffer];
    image = [image imageByCroppingToRect:cropRect];
    
    CGFloat scaleX = scaleSize.width / CGRectGetWidth(image.extent);
    CGFloat scaleY = scaleSize.height / CGRectGetHeight(image.extent);
    
    OSType type = CVPixelBufferGetPixelFormatType(pixelBuffer);
    if (type != kCVPixelFormatType_32BGRA) {
        return nil;
    }
    
    image = [image imageByApplyingTransform:CGAffineTransformMakeScale(scaleX, scaleY)];
    
    // Due to the way [CIContext:render:toCVPixelBuffer] works, we need to translate the image so the cropped section is at the origin
    image = [image imageByApplyingTransform:CGAffineTransformMakeTranslation(-image.extent.origin.x, -image.extent.origin.y)];
    
    CVPixelBufferRef output = NULL;
    
    CVPixelBufferCreate(nil,
                        CGRectGetWidth(image.extent),
                        CGRectGetHeight(image.extent),
                        CVPixelBufferGetPixelFormatType(pixelBuffer),
                        nil,
                        &output);
    
    if (output != NULL) {
        [context render:image toCVPixelBuffer:output];
    }
    
    return output;
}

- (NSData *)rgbDataFromBuffer:(CVPixelBufferRef)buffer byteCount:(NSUInteger)byteCount isModelQuantized:(BOOL)isModelQuantized {
    CVPixelBufferLockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);

    uint8_t *mutableRawPointer = (uint8_t *)CVPixelBufferGetBaseAddress(buffer);
    size_t count = CVPixelBufferGetDataSize(buffer);
    
    // 不用 bufferData 直接用 mutableRawPointer 取 byte 的話會 crash，原因不明。
    NSData *bufferData = [NSData dataWithBytesNoCopy:mutableRawPointer length:count];
    uint8_t *bytesPtr = (uint8_t *)[bufferData bytes];
    
    NSMutableData *rgbData = [[NSMutableData alloc] initWithCapacity:byteCount];
    uint8_t rgb[byteCount];
    
    NSUInteger index = 0;
    for (int i = 0; i < count; i++) {
        if (i % 4 != 3 && index < byteCount) {
            rgb[index++] = mutableRawPointer[i];
        }
    }
    [rgbData appendBytes:rgb length:byteCount];
    
    CVPixelBufferUnlockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
    
    return rgbData;
}

//- (CVPixelBufferRef)cropImage:(CVPixelBufferRef)pixelBufferRef scaleSize:(CGSize)scaleSize {
//    size_t imageWidth = CVPixelBufferGetWidth(pixelBufferRef);
//    size_t imageHeight = CVPixelBufferGetHeight(pixelBufferRef);
//    OSType pixelBufferType = CVPixelBufferGetPixelFormatType(pixelBufferRef);
//
//    if (pixelBufferType != kCVPixelFormatType_32BGRA) {
//        return nil;
//    }
//
//    size_t inputImageRowBytes = CVPixelBufferGetBytesPerRow(pixelBufferRef);
//    size_t imageChannels = 4;
//
//    size_t thumbnailSize = MIN(imageWidth, imageHeight);
//    CVPixelBufferLockBaseAddress(pixelBufferRef, 0);
//
//    CGFloat originX = 0;
//    CGFloat originY = 0;
//
//    if (imageWidth > imageHeight) {
//        originX = (imageWidth - imageHeight) / 2;
//    } else {
//        originY = (imageHeight - imageWidth) / 2;
//    }
//
//    // Finds the biggest square in the pixel buffer and advances rows based on it.
//    void *inputBaseAddress = CVPixelBufferGetBaseAddress(pixelBufferRef);
//
//    // Gets vImage Buffer from input image
//    var inputVImageBuffer = vImage_Buffer(
//                                          data: inputBaseAddress, height: UInt(thumbnailSize), width: UInt(thumbnailSize),
//                                          rowBytes: inputImageRowBytes)
//
//    size_t thumbnailRowBytes = int(size.width) * imageChannels;
//    guard  let thumbnailBytes = malloc(Int(size.height) * thumbnailRowBytes) else {
//        return nil
//    }
//
//    // Allocates a vImage buffer for thumbnail image.
//    var thumbnailVImageBuffer = vImage_Buffer(data: thumbnailBytes, height: UInt(size.height), width: UInt(size.width), rowBytes: thumbnailRowBytes)
//
//    // Performs the scale operation on input image buffer and stores it in thumbnail image buffer.
//    let scaleError = vImageScale_ARGB8888(&inputVImageBuffer, &thumbnailVImageBuffer, nil, vImage_Flags(0))
//
//    CVPixelBufferUnlockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0))
//
//    guard scaleError == kvImageNoError else {
//        return nil
//    }
//
//    let releaseCallBack: CVPixelBufferReleaseBytesCallback = {mutablePointer, pointer in
//
//        if let pointer = pointer {
//            free(UnsafeMutableRawPointer(mutating: pointer))
//        }
//    }
//
//    var thumbnailPixelBuffer: CVPixelBuffer?
//
//    // Converts the thumbnail vImage buffer to CVPixelBuffer
//    let conversionStatus = CVPixelBufferCreateWithBytes(
//                                                        nil, Int(size.width), Int(size.height), pixelBufferType, thumbnailBytes,
//                                                        thumbnailRowBytes, releaseCallBack, nil, nil, &thumbnailPixelBuffer)
//
//    guard conversionStatus == kCVReturnSuccess else {
//
//        free(thumbnailBytes)
//        return nil
//    }
//
//    return thumbnailPixelBuffer
//}

@end
