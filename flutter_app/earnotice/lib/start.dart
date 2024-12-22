import 'package:flutter/material.dart';
import 'bluetooth_screen.dart';  // Bluetooth接続確認画面をインポート

class StartScreen extends StatefulWidget {
  const StartScreen({Key? key}) : super(key: key);

  @override
  _StartScreenState createState() => _StartScreenState();
}

class _StartScreenState extends State<StartScreen> {
  @override
  void initState() {
    super.initState();
    // スプラッシュスクリーン表示後にBluetooth接続確認画面に遷移
    Future.delayed(const Duration(seconds: 3), () {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (context) => const BluetoothScreen()),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.deepPurple,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text(
              'EarNotice',
              style: TextStyle(color: Colors.white,fontSize: 20),
            ),
            Image.asset('assets/logo.png', width: 150, height: 150),
            const SizedBox(height: 20),
            const Text(
              'Loading...',
              style: TextStyle(color: Colors.white, fontSize: 18),
            ),
            const SizedBox(height: 20),
            const CircularProgressIndicator(color: Colors.white),
          ],
        ),
      ),
    );
  }
}
