import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'main.dart'; // MyHomePageへの遷移

class BluetoothScreen extends StatefulWidget {
  final FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin; // 追加

  const BluetoothScreen({Key? key, required this.flutterLocalNotificationsPlugin}) : super(key: key);

  @override
  _BluetoothScreenState createState() => _BluetoothScreenState();
}

class _BluetoothScreenState extends State<BluetoothScreen> {
  bool _isScanning = false;
  List<int>? characteristicValue;
  BluetoothCharacteristic? characteristic; // キャラクタリスティックオブジェクトを保存する変数を追加

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

  // 接続後にサービスを探索し、キャラクタリスティックのUUIDを取得する
  Future<void> _connectAndDiscoverServices(BluetoothDevice device) async {
    try {
      await device.connect();
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('${device.name} に接続しました')),
      );

      // サービスを探索
      List<BluetoothService> services = await device.discoverServices();
      for (BluetoothService service in services) {
        print('サービスUUID: ${service.uuid}');
        for (BluetoothCharacteristic characteristic in service.characteristics) {
          // 各キャラクタリスティックのUUIDとプロパティを表示
          print('キャラクタリスティックUUID: ${characteristic.uuid}');
          print('プロパティ: Read(${characteristic.properties.read}), '
              'Write(${characteristic.properties.write}), '
              'Notify(${characteristic.properties.notify})');

          // 対象のキャラクタリスティックUUIDを保存
          if (characteristic.properties.read) {
            this.characteristic = characteristic; // キャラクタリスティックを保存
            characteristicValue = await characteristic.read();
            print('キャラクタリスティックの値: $characteristicValue');
          }
        }
      }

      // キャラクタリスティック値が取得できた後に画面遷移
      if (characteristicValue != null && characteristic != null) {
        Navigator.of(context).pushReplacement(
          MaterialPageRoute(
            builder: (context) => MyHomePage(
              title: 'EarNotice -Demo-',  // タイトル
              flutterLocalNotificationsPlugin: widget.flutterLocalNotificationsPlugin,  // 通知プラグイン
              characteristic: characteristic,
              characteristicUuid: characteristic?.uuid.toString(),  // UUID を渡す
            ),
          ),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('キャラクタリスティックの値の取得に失敗しました')),
        );
      }

    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('接続中にエラーが発生しました: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Bluetooth接続確認',
          style: TextStyle(color: Colors.white, fontSize: 20),
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
                  final filteredDevices = snapshot.data!.where((result) {
                    return result.device.name == 'EarNotice';
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
                          await _connectAndDiscoverServices(device);
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
