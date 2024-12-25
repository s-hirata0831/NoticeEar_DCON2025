import 'package:flutter/material.dart';
import 'package:flite_flutter/flite_flutter.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: SpeechTestPage(),
    );
  }
}

class SpeechTestPage extends StatefulWidget {
  @override
  _SpeechTestPageState createState() => _SpeechTestPageState();
}

class _SpeechTestPageState extends State<SpeechTestPage> {
  final FliteFlutter _flite = FliteFlutter();
  String _status = "準備完了";

  @override
  void initState() {
    super.initState();
    _initializeFlite();
  }

  Future<void> _initializeFlite() async {
    try {
      await _flite.setVoice("cmu_us_kal"); // 音声を設定（適宜変更可能）
      setState(() {
        _status = "音声合成の準備ができました";
      });
    } catch (e) {
      setState(() {
        _status = "エラー: ${e.toString()}";
      });
    }
  }

  void _speak(String text) async {
    setState(() {
      _status = "音声再生中...";
    });

    try {
      await _flite.speak(text);
      setState(() {
        _status = "音声再生完了";
      });
    } catch (e) {
      setState(() {
        _status = "エラー: ${e.toString()}";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Flite Flutter テスト"),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              _status,
              style: TextStyle(fontSize: 18),
            ),
            SizedBox(height: 20),
            TextField(
              decoration: InputDecoration(
                border: OutlineInputBorder(),
                labelText: "テキストを入力してください",
              ),
              onSubmitted: (value) {
                if (value.isNotEmpty) {
                  _speak(value);
                }
              },
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                _speak("こんにちは、これはFlite Flutterのテストです。");
              },
              child: Text("テスト音声を再生"),
            ),
          ],
        ),
      ),
    );
  }
}
