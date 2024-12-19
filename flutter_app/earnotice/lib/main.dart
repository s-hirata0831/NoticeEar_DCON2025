import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:flutter_sound/flutter_sound.dart'; // flutter_soundのインポート
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';// flutter_blue_plusのインポート

void main() {
  WidgetsFlutterBinding.ensureInitialized();

  FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin =
  FlutterLocalNotificationsPlugin();

  // Android 用通知の初期化
  const initializationSettingsAndroid =
  AndroidInitializationSettings('@mipmap/ic_launcher');
  const initializationSettings =
  InitializationSettings(android: initializationSettingsAndroid);

  flutterLocalNotificationsPlugin.initialize(initializationSettings);

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  int _currentIndex = 0; // 現在のナビゲーションバーのインデックス
  final List<Map<String, String>> _notificationHistory = []; // 通知履歴リスト

  final FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin =
  FlutterLocalNotificationsPlugin();

  final FlutterSoundRecorder _recorder = FlutterSoundRecorder(); // flutter_soundのRecorder
  bool _isRecording = false; // 録音中かどうかのフラグ

  @override
  void initState() {
    super.initState();
    _initRecorder();
  }

  // Recorderの初期化
  Future<void> _initRecorder() async {
    // マイク権限をリクエスト
    if (await Permission.microphone.request().isGranted) {
      // 録音の準備
      await _recorder.openRecorder();
    } else {
      throw Exception('マイクの権限がありません');
    }
  }

  // 録音の開始・停止を切り替え
  void _toggleRecording() async {
    if (_isRecording) {
      // 録音停止
      await _recorder.stopRecorder();
    } else {
      // 録音開始
      await _recorder.startRecorder(toFile: 'audio_example.aac');
    }

    // 状態を更新
    setState(() {
      _isRecording = !_isRecording;
    });
  }

  @override
  void dispose() {
    super.dispose();
    _recorder.closeRecorder(); // 録音セッションを閉じる
  }

  // 通知を表示
  void _showNotification() async {
    final currentTime = DateTime.now();
    final formattedTime =
        "${currentTime.hour}:${currentTime.minute}:${currentTime.second}";

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

  // _showSnackbar メソッドをクラス内に追加
  void _showSnackbar(BuildContext context, String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
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
              onTap: _toggleRecording, // 録音の開始・停止を切り替え
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
      // 設定画面
      Center(
        child: Text("非通知リスト"),
      ),
      Center(
        child: Text("設定画面"),
      ),
      Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // BLEスキャン時の処理
            ElevatedButton(
              onPressed: () {
                // BLEスキャンを開始
                FlutterBluePlus.startScan(timeout: const Duration(seconds: 5));
              },
              child: const Text('デバイスをスキャン'),
            ),
            Expanded(
              child: StreamBuilder<List<ScanResult>>(
                stream: FlutterBluePlus.scanResults,
                builder: (context, snapshot) {
                  if (snapshot.connectionState == ConnectionState.waiting) {
                    return const Center(child: CircularProgressIndicator());
                  }
                  if (snapshot.hasData) {
                    final devices = snapshot.data!;
                    return ListView.builder(
                      itemCount: devices.length,
                      itemBuilder: (context, index) {
                        final device = devices[index].device;
                        return ListTile(
                          title: Text(device.name.isEmpty ? '未知のデバイス' : device.name),
                          subtitle: Text(device.id.toString()),
                          onTap: () async {
                            await device.connect();
                            _showSnackbar(context, '${device.name} に接続しました');
                            final services = await device.discoverServices();
                            _showSnackbar(context, '${services.length} サービス発見');
                          },
                        );
                      },
                    );
                  }
                  return const Center(child: Text('デバイスが見つかりませんでした'));
                },
              ),
            ),
          ],
        ),
      ),

    ];

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'EarNotice-Demo-',
          style: TextStyle(
            color: Colors.white,
            fontSize: 20,
          ),
        ),
        backgroundColor: Colors.deepPurpleAccent,
        actions: [
          IconButton(
            icon: const Icon(Icons.notifications_active, color: Colors.white),
            onPressed: _showNotification, // 通知を表示するメソッドを呼び出し
          ),
          const Icon(Icons.share, color: Colors.yellow),
        ],
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
            label: '非通知リスト',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.settings),
            label: '設定',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.bluetooth),
            label: 'Bluetooth',
          ),
        ],
      ),
    );
  }
}
