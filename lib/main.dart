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
  late Interpreter interpreter;
  late List<String> labels;
  bool isModelLoaded = false;
  String predictedSign = "";
  bool isProcessing = false;

@override
void initState() {
  super.initState();

  _controller = CameraController(
    cameras[0],
    ResolutionPreset.medium,
    enableAudio: false,
  );

  _controller.initialize().then((_) async {
    if (!mounted) return;

    // Load model and labels first
    await loadModelAndLabels();

    // Now start streaming frames
    _controller.startImageStream((CameraImage image) {
      // Debug print to confirm preprocessing is running
      print("Frame received: ${image.width}x${image.height}");

      if (!isProcessing) {
        isProcessing = true;
        predict(image).then((_) => isProcessing = false);
      }
    });

    setState(() {});
  });
}


  Future<void> loadModelAndLabels() async {
    try {
      interpreter = await Interpreter.fromAsset('assets/bdslw_model.tflite');
      final labelsData = await rootBundle.loadString('assets/labels.txt');
      labels = labelsData.split('\n').map((e) => e.trim()).toList();
      isModelLoaded = true;
      print("Model and labels loaded successfully!");
    } catch (e) {
      print("Error loading model: $e");
    }
  }

  // Convert YUV420 to RGB and preprocess
  List<List<List<List<double>>>> preprocessCameraImage(CameraImage image) {
    final width = image.width;
    final height = image.height;

    print("Preprocessing camera frame: ${image.width}x${image.height}");


    // YUV420 to RGB conversion
    Uint8List y = image.planes[0].bytes;
    Uint8List u = image.planes[1].bytes;
    Uint8List v = image.planes[2].bytes;

    List<int> rgbBytes = List.filled(width * height * 3, 0);

    int uvRowStride = image.planes[1].bytesPerRow;
    int uvPixelStride = image.planes[1].bytesPerPixel!;

    for (int j = 0; j < height; j++) {
      for (int i = 0; i < width; i++) {
        int uvIndex = (j ~/ 2) * uvRowStride + (i ~/ 2) * uvPixelStride;
        int yIndex = j * width + i;

        int Y = y[yIndex];
        int U = u[uvIndex];
        int V = v[uvIndex];

        int R = (Y + 1.402 * (V - 128)).round().clamp(0, 255);
        int G = (Y - 0.344136 * (U - 128) - 0.714136 * (V - 128)).round().clamp(0, 255);
        int B = (Y + 1.772 * (U - 128)).round().clamp(0, 255);

        rgbBytes[yIndex * 3] = R;
        rgbBytes[yIndex * 3 + 1] = G;
        rgbBytes[yIndex * 3 + 2] = B;
      }
    }

    // Crop center square
    int cropSize = min(width, height);
    int offsetX = (width - cropSize) ~/ 2;
    int offsetY = (height - cropSize) ~/ 2;

    List<int> croppedRgb = List.filled(cropSize * cropSize * 3, 0);
    for (int yC = 0; yC < cropSize; yC++) {
      for (int xC = 0; xC < cropSize; xC++) {
        int srcIndex = ((yC + offsetY) * width + (xC + offsetX)) * 3;
        int dstIndex = (yC * cropSize + xC) * 3;
        croppedRgb[dstIndex] = rgbBytes[srcIndex];
        croppedRgb[dstIndex + 1] = rgbBytes[srcIndex + 1];
        croppedRgb[dstIndex + 2] = rgbBytes[srcIndex + 2];
      }
    }

    // Create img.Image
    final imgData = img.Image.fromBytes(
      width: cropSize,
      height: cropSize,
      bytes: Uint8List.fromList(croppedRgb).buffer,
      numChannels: 3,
      format: img.Format.uint8,
    );

    // Resize to 224x224
    final resized = img.copyResize(imgData, width: 224, height: 224);

    // Normalize and reshape to [1,224,224,3]
    var input = List.generate(
      1,
      (_) => List.generate(
        224,
        (_) => List.generate(224, (_) => List.filled(3, 0.0)),
      ),
    );

    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        final pixel = resized.getPixel(x, y);
        input[0][y][x][0] = pixel.r / 255.0;
        input[0][y][x][1] = pixel.g / 255.0;
        input[0][y][x][2] = pixel.b / 255.0;
      }
    }

    return input;
  }

  // Run prediction
  Future<void> predict(CameraImage image) async {
    var input = preprocessCameraImage(image);
    var output = List.generate(1, (_) => List.filled(labels.length, 0.0));

    interpreter.run(input, output);

    double maxValue = output[0][0];
    int maxIndex = 0;
    for (int i = 1; i < labels.length; i++) {
      if (output[0][i] > maxValue) {
        maxValue = output[0][i];
        maxIndex = i;
      }
    }

    setState(() {
      predictedSign = labels[maxIndex];
    });
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      body: Stack(
        children: [
          CameraPreview(_controller),
          // Red square overlay
          Center(
            child: Container(
              width: 224,
              height: 224,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.red, width: 2),
              ),
            ),
          ),
          // Predicted sign
          Positioned(
            bottom: 50,
            left: 20,
            child: Container(
              padding: const EdgeInsets.all(10),
              color: Colors.black54,
              child: Text(
                predictedSign,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
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
    _controller.dispose();
    interpreter.close();
    super.dispose();
  }
}

