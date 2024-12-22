import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'start.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();

  FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin =
  FlutterLocalNotificationsPlugin();

  const initializationSettingsAndroid =
  AndroidInitializationSettings('@mipmap/ic_launcher');
  const initializationSettings =
  InitializationSettings(android: initializationSettingsAndroid);

  flutterLocalNotificationsPlugin.initialize(initializationSettings);

  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const StartScreen(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  final String title;

  const MyHomePage({super.key, required this.title});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  int _currentIndex = 0;
  final List<Map<String, String>> _notificationHistory = [];
  final FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin =
  FlutterLocalNotificationsPlugin();
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  bool _isRecording = false;
  bool _isScanning = false;

  @override
  void initState() {
    super.initState();
    _initRecorder();
  }

  // 録音の初期化
  Future<void> _initRecorder() async {
    if (await Permission.microphone.request().isGranted) {
      await _recorder.openRecorder();
    } else {
      throw Exception('マイクの権限がありません');
    }
  }

  // 録音のトグル
  void _toggleRecording() async {
    if (_isRecording) {
      await _recorder.stopRecorder();
    } else {
      await _recorder.startRecorder(toFile: 'audio_example.aac');
    }
    setState(() {
      _isRecording = !_isRecording;
    });
  }

  @override
  void dispose() {
    super.dispose();
    _recorder.closeRecorder();
  }

  void _showNotification() async {
    final currentTime = DateTime.now();
    final formattedTime = "${currentTime.hour}:${currentTime.minute}:${currentTime.second}";

    setState(() {
      _notificationHistory.add({
        "title": "Push通知テスト中",
        "message": "テキスト変更：見えてますか？",
        "time": formattedTime,
      });
    });

    const androidNotificationDetail = AndroidNotificationDetails(
      'channel_id',
      'channel_name',
    );
    const notificationDetail = NotificationDetails(
      android: androidNotificationDetail,
    );

    await flutterLocalNotificationsPlugin.show(
      0,
      'Push通知テスト中',
      'テキスト変更：見えてますか？',
      notificationDetail,
    );
  }

  void _showSnackbar(BuildContext context, String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  // Bluetoothスキャンの開始/停止
  void _startScan() {
    if (!_isScanning) {
      setState(() {
        _isScanning = true;
      });
      FlutterBluePlus.startScan(timeout: const Duration(seconds: 5));
    } else {
      setState(() {
        _isScanning = false;
      });
      FlutterBluePlus.stopScan();
    }
  }

  @override
  Widget build(BuildContext context) {
    final List<Widget> pages = [
      // ホーム画面
      Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            GestureDetector(
              onTap: _toggleRecording,
              child: CircleAvatar(
                radius: 70.0,
                backgroundColor: _isRecording ? Colors.red : Colors.green,
                child: Transform.scale(
                  scale: 1.5,
                  child: Icon(
                    _isRecording ? Icons.stop : Icons.mic,
                    size: 40.0,
                    color: Colors.white,
                  ),
                ),
              ),
            ),
            SizedBox(height: 20),
            Text(
              _isRecording ? '録音中...' : '録音停止中',
              style: TextStyle(fontSize: 18.0),
            ),
          ],
        ),
      ),
      // 通知履歴画面
      Center(
        child: _notificationHistory.isEmpty
            ? const Text('通知履歴はありません')
            : ListView.builder(
          itemCount: _notificationHistory.length,
          itemBuilder: (context, index) {
            final history = _notificationHistory[index];
            return ListTile(
              title: Text(history["title"] ?? ""),
              subtitle: Text(history["message"] ?? ""),
              trailing: Text(history["time"] ?? ""),
            );
          },
        ),
      ),
      Center(
        child: Text("非表示リスト"),
      ),
      // 設定画面
      Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: const [
            Icon(Icons.settings, size: 50),
            SizedBox(height: 20),
            Text("設定画面", style: TextStyle(fontSize: 24)),
          ],
        ),
      ),
    ];

    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
        backgroundColor: Colors.deepPurpleAccent,
        actions: [
          IconButton(
            icon: const Icon(Icons.notifications_active, color: Colors.white),
            onPressed: _showNotification,
          ),
        ],
        titleTextStyle: TextStyle(
          color: Colors.white,
          fontSize: 20,
        ),
      ),
      body: pages[_currentIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
        selectedItemColor: Colors.black,
        unselectedItemColor: Colors.black,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            label: 'ホーム',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.history),
            label: '通知履歴',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.playlist_remove),
            label: '非表示リスト',
          ),
          BottomNavigationBarItem(
              icon: Icon(Icons.settings),
              label: '設定'
          ),
        ],
      ),
    );
  }
}
