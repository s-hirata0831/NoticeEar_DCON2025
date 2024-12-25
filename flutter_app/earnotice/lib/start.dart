import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';  // 追加
import 'bluetooth_screen.dart';  // Bluetooth接続確認画面をインポート

class StartScreen extends StatefulWidget {
  const StartScreen({Key? key}) : super(key: key);

  @override
  _StartScreenState createState() => _StartScreenState();
}

class _StartScreenState extends State<StartScreen> {
  late FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin;  // 追加

  @override
  void initState() {
    super.initState();
    flutterLocalNotificationsPlugin = FlutterLocalNotificationsPlugin();  // 初期化

    // 通知の初期化処理
    _initializeNotifications();

    // スプラッシュスクリーン表示後にBluetooth接続確認画面に遷移
    Future.delayed(const Duration(seconds: 3), () {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (context) => BluetoothScreen(
            flutterLocalNotificationsPlugin: flutterLocalNotificationsPlugin,  // 通知プラグインを渡す
          ),
        ),
      );
    });
  }

  // 非同期で通知プラグインの初期化を行う
  Future<void> _initializeNotifications() async {
    final initializationSettings = InitializationSettings(
      android: AndroidInitializationSettings('app_icon'),
    );
    await flutterLocalNotificationsPlugin.initialize(initializationSettings);
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
              style: TextStyle(color: Colors.white, fontSize: 20),
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
