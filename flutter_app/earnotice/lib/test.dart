import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: CounterScreen(),
    );
  }
}

class CounterScreen extends StatefulWidget {
  @override
  _CounterScreenState createState() => _CounterScreenState();
}

class _CounterScreenState extends State<CounterScreen> {
  // カウントを保持する変数
  int _counter = 0;

  // カウントを増やす関数
  void _incrementCounter() {
    setState(() {
      _counter++;
    });
  }

  // カウントをリセットする関数
  void _resetCounter() {
    setState(() {
      _counter = 0;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('カウンターアプリ'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Text(
              '現在のカウント:',
              style: TextStyle(fontSize: 20),
            ),
            Text(
              '$_counter', // カウントの表示
              style: TextStyle(fontSize: 50, fontWeight: FontWeight.bold),
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                ElevatedButton(
                  onPressed: _incrementCounter, // 増加ボタン
                  child: Text('増加'),
                ),
                SizedBox(width: 20), // ボタン間のスペース
                ElevatedButton(
                  onPressed: _resetCounter, // リセットボタン
                  child: Text('リセット'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}