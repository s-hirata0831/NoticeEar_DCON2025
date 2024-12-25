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

  // 通知チャネルを作成（デフォルトの通知音を使用）
  const androidNotificationChannel = AndroidNotificationChannel(
    'default_channel_id', // チャネルID
    'Default Channel', // チャネル名
    description: 'This is the default notification channel',
    importance: Importance.high,
    playSound: true, // 音を再生
  );

  flutterLocalNotificationsPlugin
      .resolvePlatformSpecificImplementation<AndroidFlutterLocalNotificationsPlugin>()
      ?.createNotificationChannel(androidNotificationChannel);

  runApp(MyApp(flutterLocalNotificationsPlugin: flutterLocalNotificationsPlugin));
}

class MyApp extends StatelessWidget {
  final FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin;

  const MyApp({super.key, required this.flutterLocalNotificationsPlugin});

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
  final FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin;
  final BluetoothCharacteristic? characteristic;
  final String? characteristicUuid;

  const MyHomePage({
    Key? key,
    required this.title,
    required this.flutterLocalNotificationsPlugin,
    required this.characteristic,
    required this.characteristicUuid,
  }): super(key: key);

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  int _currentIndex = 0;
  final List<Map<String, String>> _notificationHistory = [];
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  bool _isRecording = false;
  BluetoothCharacteristic? characteristicUuid;

  String _characteristicValue = "No Data";

  @override
  void initState() {
    super.initState();
    _initRecorder();
    _readCharacteristic();
  }

  // Initialize the recorder with microphone permission
  Future<void> _initRecorder() async {
    if (await Permission.microphone.request().isGranted) {
      await _recorder.openRecorder();
    } else {
      throw Exception('マイクの権限がありません');
    }
  }

  Future<void> _readCharacteristic() async {
    if (widget.characteristic != null) {
      try {
        final value = await widget.characteristic!.read(); // キャラクタリスティックの値を読み込む
        setState(() {
          _characteristicValue = String.fromCharCodes(value); // バイト値を文字列に変換
        });
      } catch (e) {
        print("Error reading characteristic: $e");
      }
    }
  }

  Future<void> _writeToCharacteristic(String data) async {
    if (widget.characteristic != null) {  // _characteristic は BluetoothCharacteristic 型であるべき
      try {
        // String をバイト列に変換して送信
        await widget.characteristic!.write(data.codeUnits, withoutResponse: false);  // withoutResponse を false に変更
        print("Data sent to characteristic: $data");
      } catch (e) {
        print("Error sending data: $e");
      }
    } else {
      print("Characteristic is null.");
    }
  }



  // Start/Stop audio recording
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

  // Clean up resources
  @override
  void dispose() {
    _recorder.closeRecorder();
    super.dispose();
  }

  // Send a notification and add it to the history
  void _sendNotification({required String title, required String message}) async {
    final currentTime = DateTime.now();
    final formattedTime = "${currentTime.hour}:${currentTime.minute}:${currentTime.second}";

    setState(() {
      _notificationHistory.add({
        "title": title,
        "message": message,
        "time": formattedTime,
      });
    });

    const androidNotificationDetail = AndroidNotificationDetails(
      'default_channel',
      'Default Channel',
    );

    const notificationDetail = NotificationDetails(
      android: androidNotificationDetail,
    );

    try {
      await widget.flutterLocalNotificationsPlugin.show(
        currentTime.second,
        title,
        message,
        notificationDetail,
      );
      print("通知を送信しました");
    } catch (e) {
      print("通知送信エラー: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    final List<Widget> pages = [
      // Home screen
      Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            GestureDetector(
              onTap: _toggleRecording,
              child: CircleAvatar(
                radius: 70.0,
                backgroundColor: _isRecording ? Colors.red : Colors.green,
                child: Icon(
                  _isRecording ? Icons.stop : Icons.mic,
                  size: 40.0,
                  color: Colors.white,
                ),
              ),
            ),
            SizedBox(height: 20),
            Text(
              _isRecording ? '録音中...' : '録音停止中',
              style: TextStyle(fontSize: 20.0),
            ),
          ],
        ),
      ),
      // Notification history screen
      Center(
        child: _notificationHistory.isEmpty
            ? Text('まだ通知はありません')
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
      // Non-notified list screen
      Center(
        child: Text("非通知リスト"),
      ),
      // Settings screen
      Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.settings, size: 50),
            SizedBox(height: 20),
            Text("設定", style: TextStyle(fontSize: 24)),
          ],
        ),
      ),
    ];

    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title, style:TextStyle(color: Colors.white)),
        backgroundColor: Colors.deepPurpleAccent,
        actions: [
          IconButton(
            icon: Icon(Icons.notifications_active, color: Colors.white),
            onPressed: () {
              _writeToCharacteristic("common");
              _sendNotification(title: '音声検出：電子レンジ', message: '電子レンジの音が鳴りました');
            },
          ),
          IconButton(
            icon: Icon(Icons.pets, color: Colors.white),
            onPressed: () {
              _writeToCharacteristic("animal");
              _sendNotification(title: '音声検出：動物', message: '動物が鳴きました');
            },
          ),
          IconButton(
            icon: Icon(Icons.directions_car, color: Colors.white),
            onPressed: () {
              _writeToCharacteristic("emerge");
              _sendNotification(title: '音声検出：クラクション', message: 'クラクションが鳴りました');
            },
          ),
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
        unselectedItemColor: Colors.grey,
        items: [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: 'ホーム'),
          BottomNavigationBarItem(icon: Icon(Icons.history), label: '通知履歴'),
          BottomNavigationBarItem(icon: Icon(Icons.playlist_remove), label: '非通知リスト'),
          BottomNavigationBarItem(icon: Icon(Icons.settings), label: '設定'),
        ],
      ),
    );
  }
}