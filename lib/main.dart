import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'dart:typed_data';
import 'dart:math';

late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: CameraScreen(),
    );
  }
}

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController _controller;
  Interpreter? interpreter;
  List<String> labels = [];

  bool isModelLoaded = false;
  bool isProcessing = false;
  String predictedSign = "Initializing...";
  double confidence = 0.0;

  int _lastRun = 0;
  
  // Reusable buffers to reduce memory allocation
  late List<int> _rgbBuffer;
  late List<List<List<List<double>>>> _inputBuffer;

  @override
  void initState() {
    super.initState();
    print("🚀 App started");
    _initializeBuffers();
    _initializeCamera();
  }

  void _initializeBuffers() {
    print("📦 Initializing buffers...");
    const int maxCropSize = 1920;
    _rgbBuffer = List.filled(maxCropSize * maxCropSize * 3, 0);
    
    _inputBuffer = List.generate(
      1,
      (_) => List.generate(
        224,
        (_) => List.generate(224, (_) => List.filled(3, 0.0)),
      ),
    );
    print("✅ Buffers initialized");
  }

  void _initializeCamera() {
    print("📷 Initializing camera...");
    
    // Use back camera for better quality
    final camera = cameras.firstWhere(
      (cam) => cam.lensDirection == CameraLensDirection.back,
      orElse: () => cameras[0],
    );
    
    print("📷 Selected camera: ${camera.lensDirection}");
    
    _controller = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    _controller.initialize().then((_) async {
      if (!mounted) return;

      print("✅ Camera initialized: ${_controller.value.previewSize}");
      print("📐 Aspect ratio: ${_controller.value.aspectRatio}");
      
      await loadModelAndLabels();

      _controller.startImageStream((CameraImage image) {
        final now = DateTime.now().millisecondsSinceEpoch;

        if (isModelLoaded && !isProcessing && now - _lastRun > 500) {
          _lastRun = now;
          isProcessing = true;

          predict(image).catchError((e) {
            print("❌ Prediction error: $e");
          }).whenComplete(() {
            isProcessing = false;
          });
        }
      });

      setState(() {});
    }).catchError((e) {
      print("❌ Camera initialization error: $e");
      setState(() {
        predictedSign = "Camera Error: Check permissions";
      });
    });
  }

  Future<void> loadModelAndLabels() async {
    // TEMPORARY FIX: Hard-code labels instead of loading from file
    print("⚠️ Using hard-coded labels (temporary fix)");
    labels = [
      'Bad',
      'Beautiful', 
      'Friend',
      'Good',
      'House',
      'Me',
      'My',
      'Request',
      'Skin',
      'Urine',
      'You'
    ];
    print("✅ Hard-coded ${labels.length} labels: $labels");

    // Now try loading model
    try {
      print("🔄 Attempting to load model...");
      print("📁 Looking for: assets/bdslw_model.tflite");
      
      interpreter = await Interpreter.fromAsset('assets/bdslw_model.tflite').timeout(
        const Duration(seconds: 10),
        onTimeout: () {
          throw Exception("Timeout loading model - file may not exist");
        },
      );
      
      print("✅ Model file loaded!");
      
      interpreter!.allocateTensors();
      print("✅ Tensors allocated");
      
      final inputTensor = interpreter!.getInputTensor(0);
      final outputTensor = interpreter!.getOutputTensor(0);
      print("📊 Input shape: ${inputTensor.shape}");
      print("📊 Output shape: ${outputTensor.shape}");

      isModelLoaded = true;
      setState(() {
        predictedSign = "Ready - ${labels.length} signs";
      });
      print("🎉 SUCCESS! Model ready with hard-coded labels!");
      
    } catch (e, stackTrace) {
      print("❌ MODEL LOADING FAILED");
      print("Error: $e");
      print("Stack trace: $stackTrace");
      setState(() {
        predictedSign = "Can't find bdslw_model.tflite in assets folder";
      });
    }
  }

  void preprocessCameraImage(CameraImage image) {
    try {
      final int cropSize = min(image.width, image.height);
      final int offsetX = (image.width - cropSize) ~/ 2;
      final int offsetY = (image.height - cropSize) ~/ 2;

      final Uint8List yPlane = image.planes[0].bytes;
      final Uint8List uPlane = image.planes[1].bytes;
      final Uint8List vPlane = image.planes[2].bytes;
      
      final int yRowStride = image.planes[0].bytesPerRow;
      final int uvRowStride = image.planes[1].bytesPerRow;
      final int uvPixelStride = 2;

      // Convert YUV to RGB and crop
      for (int y = 0; y < cropSize; y++) {
        for (int x = 0; x < cropSize; x++) {
          final int yIndex = (y + offsetY) * yRowStride + (x + offsetX);
          final int uvIndex = ((y + offsetY) ~/ 2) * uvRowStride + 
                              ((x + offsetX) ~/ 2) * uvPixelStride;
          
          final int yValue = yPlane[yIndex];
          final int uValue = uvIndex < uPlane.length ? uPlane[uvIndex] : 128;
          final int vValue = uvIndex < vPlane.length ? vPlane[uvIndex] : 128;
          
          // YUV to RGB conversion
          int r = (yValue + 1.402 * (vValue - 128)).round().clamp(0, 255);
          int g = (yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128))
              .round().clamp(0, 255);
          int b = (yValue + 1.772 * (uValue - 128)).round().clamp(0, 255);
          
          final int idx = (y * cropSize + x) * 3;
          _rgbBuffer[idx] = r;
          _rgbBuffer[idx + 1] = g;
          _rgbBuffer[idx + 2] = b;
        }
      }

      final img.Image imgData = img.Image.fromBytes(
        width: cropSize,
        height: cropSize,
        bytes: Uint8List.fromList(_rgbBuffer.sublist(0, cropSize * cropSize * 3)).buffer,
        numChannels: 3,
        format: img.Format.uint8,
      );

      final img.Image resized = img.copyResize(
        imgData,
        width: 224,
        height: 224,
        interpolation: img.Interpolation.linear,
      );

      // Normalize to 0-1 (matching training)
      for (int y = 0; y < 224; y++) {
        for (int x = 0; x < 224; x++) {
          final pixel = resized.getPixel(x, y);
          _inputBuffer[0][y][x][0] = pixel.r / 255.0;
          _inputBuffer[0][y][x][1] = pixel.g / 255.0;
          _inputBuffer[0][y][x][2] = pixel.b / 255.0;
        }
      }
    } catch (e) {
      print("❌ Preprocessing error: $e");
      rethrow;
    }
  }

  Future<void> predict(CameraImage image) async {
    if (interpreter == null || labels.isEmpty) {
      print("⚠️ Model not ready - skipping prediction");
      return;
    }

    try {
      preprocessCameraImage(image);

      final output = List.generate(1, (_) => List.filled(labels.length, 0.0));

      interpreter!.run(_inputBuffer, output);

      double maxVal = output[0][0];
      int maxIndex = 0;

      for (int i = 1; i < labels.length; i++) {
        if (output[0][i] > maxVal) {
          maxVal = output[0][i];
          maxIndex = i;
        }
      }

      // Show top 3 predictions
      final predictions = <MapEntry<int, double>>[];
      for (int i = 0; i < labels.length; i++) {
        predictions.add(MapEntry(i, output[0][i]));
      }
      predictions.sort((a, b) => b.value.compareTo(a.value));
      
      print("🔍 Top 3: ${predictions.take(3).map((e) => 
        "${labels[e.key]} (${(e.value * 100).toStringAsFixed(1)}%)").join(", ")}");

      if (maxVal > 0.2) {
        setState(() {
          predictedSign = labels[maxIndex];
          confidence = maxVal;
        });
      } else {
        setState(() {
          predictedSign = "No clear sign";
          confidence = maxVal;
        });
      }

    } catch (e) {
      print("❌ Prediction error: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return Scaffold(
        backgroundColor: Colors.black,
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const CircularProgressIndicator(color: Colors.greenAccent),
              const SizedBox(height: 20),
              Text(
                predictedSign,
                style: const TextStyle(color: Colors.white, fontSize: 16),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // Camera Preview
          Center(
            child: AspectRatio(
              aspectRatio: _controller.value.aspectRatio,
              child: CameraPreview(_controller),
            ),
          ),

          // Guide square
          Center(
            child: Container(
              width: 280,
              height: 280,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.greenAccent, width: 3),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 6,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.greenAccent.withOpacity(0.3),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: const Text(
                      "Position hand here",
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 14,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                ],
              ),
            ),
          ),

          // Prediction display
          Positioned(
            bottom: 40,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 24,
                  vertical: 16,
                ),
                margin: const EdgeInsets.symmetric(horizontal: 20),
                decoration: BoxDecoration(
                  color: Colors.black87,
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(
                    color: confidence > 0.6 ? Colors.greenAccent : 
                           confidence > 0.3 ? Colors.orangeAccent : Colors.red,
                    width: 2,
                  ),
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      predictedSign,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 28,
                        fontWeight: FontWeight.bold,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    if (confidence > 0)
                      Padding(
                        padding: const EdgeInsets.only(top: 8),
                        child: Column(
                          children: [
                            Text(
                              "${(confidence * 100).toStringAsFixed(1)}% confident",
                              style: TextStyle(
                                color: Colors.greenAccent.withOpacity(0.8),
                                fontSize: 14,
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              confidence > 0.6 ? "✓ Good!" : 
                              confidence > 0.3 ? "⏳ Hold steady" : "⚠ Try again",
                              style: TextStyle(
                                color: Colors.white.withOpacity(0.6),
                                fontSize: 12,
                              ),
                            ),
                          ],
                        ),
                      ),
                  ],
                ),
              ),
            ),
          ),

          // Status indicator (top right)
          Positioned(
            top: 50,
            right: 20,
            child: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: isModelLoaded ? Colors.green : Colors.red,
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: (isModelLoaded ? Colors.green : Colors.red).withOpacity(0.5),
                    blurRadius: 8,
                    spreadRadius: 2,
                  ),
                ],
              ),
              child: Icon(
                isModelLoaded ? Icons.check : Icons.close,
                color: Colors.white,
                size: 16,
              ),
            ),
          ),

          // Class count indicator (top left)
          Positioned(
            top: 50,
            left: 20,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.black87,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: Colors.greenAccent, width: 1),
              ),
              child: Text(
                labels.isEmpty ? "Loading..." : "${labels.length} signs",
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    print("🛑 Disposing resources...");
    _controller.dispose();
    interpreter?.close();
    super.dispose();
  }
}