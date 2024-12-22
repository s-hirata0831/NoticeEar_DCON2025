import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'main.dart'; // MyHomePageへの遷移

class BluetoothScreen extends StatefulWidget {
  const BluetoothScreen({Key? key}) : super(key: key);

  @override
  _BluetoothScreenState createState() => _BluetoothScreenState();
}

class _BluetoothScreenState extends State<BluetoothScreen> {
  bool _isScanning = false;

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
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Bluetooth接続確認',
          style: TextStyle(color: Colors.white,fontSize: 20),
        ),
        backgroundColor: Colors.deepPurpleAccent,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: _startScan,
              child: Text(_isScanning ? 'スキャン停止' : 'デバイスをスキャン'),
            ),
            StreamBuilder<List<ScanResult>>(
              stream: FlutterBluePlus.scanResults,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return const Center(child: CircularProgressIndicator());
                }
                if (snapshot.hasData && snapshot.data!.isNotEmpty) {
                  // スキャン結果から特定のデバイス名でフィルタリング
                  final filteredDevices = snapshot.data!.where((result) {
                    return result.device.name == 'EarNotice'; // フィルタリング条件（デバイス名）
                  }).toList();

                  if (filteredDevices.isEmpty) {
                    return const Text('対象のデバイスが見つかりませんでした');
                  }

                  final device = filteredDevices.first.device;

                  return Column(
                    children: [
                      ListTile(
                        title: Text(device.name.isEmpty ? '未知のデバイス' : device.name),
                        subtitle: Text(device.id.toString()),
                      ),
                      ElevatedButton(
                        onPressed: () async {
                          await device.connect();
                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(content: Text('${device.name} に接続しました')),
                          );
                          // 接続後にMyHomePageに遷移
                          Navigator.of(context).pushReplacement(
                            MaterialPageRoute(builder: (context) => const MyHomePage(title: 'Flutter Demo Home Page')),
                          );
                        },
                        child: const Text('接続'),
                      ),
                    ],
                  );
                }
                return const Text('デバイスが見つかりませんでした');
              },
            ),
          ],
        ),
      ),
    );
  }
}
