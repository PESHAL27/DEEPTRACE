| # | Sample ID | Filename | SHA-256 (Prefix) | Ground Truth | Predicted | Fake Score | Correct | Margin Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `EXT-REAL-0000` | `real_00000.jpg` | `ee7d2146d181...` | **AUTHENTIC** | `real` | `0.4688` | ✅ TRUE | ⚠️ Low Margin |
| 2 | `EXT-FAKE-0000` | `fake_00000.jpg` | `c69d7dc0e14f...` | **MANIPULATED** | `fake` | `0.8305` | ✅ TRUE | Standard |
| 3 | `EXT-REAL-0001` | `real_00001.jpg` | `d02ac85dbf48...` | **AUTHENTIC** | `fake` | `0.7043` | ❌ FALSE | Standard |
| 4 | `EXT-FAKE-0001` | `fake_00001.jpg` | `e0d3bd8a01b4...` | **MANIPULATED** | `real` | `0.0666` | ❌ FALSE | Standard |
| 5 | `EXT-REAL-0002` | `real_00002.jpg` | `37d5805bbfd1...` | **AUTHENTIC** | `real` | `0.2116` | ✅ TRUE | Standard |
| 6 | `EXT-FAKE-0002` | `fake_00002.jpg` | `33a0aa3be4a1...` | **MANIPULATED** | `real` | `0.4366` | ❌ FALSE | ⚠️ Low Margin |
| 7 | `EXT-REAL-0003` | `real_00003.jpg` | `324af7cdbf23...` | **AUTHENTIC** | `real` | `0.4106` | ✅ TRUE | ⚠️ Low Margin |
| 8 | `EXT-FAKE-0003` | `fake_00003.jpg` | `102f4330e5e2...` | **MANIPULATED** | `real` | `0.4011` | ❌ FALSE | ⚠️ Low Margin |
| 9 | `EXT-REAL-0004` | `real_00004.jpg` | `ee16c9e0e87a...` | **AUTHENTIC** | `real` | `0.3266` | ✅ TRUE | Standard |
| 10 | `EXT-FAKE-0004` | `fake_00004.jpg` | `7a5a1d69b316...` | **MANIPULATED** | `fake` | `0.7340` | ✅ TRUE | Standard |
| 11 | `EXT-REAL-0005` | `real_00005.jpg` | `86d2a72d89ff...` | **AUTHENTIC** | `real` | `0.3254` | ✅ TRUE | Standard |
| 12 | `EXT-FAKE-0005` | `fake_00005.jpg` | `b65cf79b9f9e...` | **MANIPULATED** | `fake` | `0.9040` | ✅ TRUE | Standard |
| 13 | `EXT-REAL-0006` | `real_00006.jpg` | `b36ba3cf9269...` | **AUTHENTIC** | `real` | `0.2173` | ✅ TRUE | Standard |
| 14 | `EXT-FAKE-0006` | `fake_00006.jpg` | `60177b92d365...` | **MANIPULATED** | `fake` | `0.6481` | ✅ TRUE | Standard |
| 15 | `EXT-REAL-0007` | `real_00007.jpg` | `dbe0dac44a1d...` | **AUTHENTIC** | `real` | `0.1351` | ✅ TRUE | Standard |
| 16 | `EXT-FAKE-0007` | `fake_00007.jpg` | `861fd2a1c68d...` | **MANIPULATED** | `real` | `0.1003` | ❌ FALSE | Standard |
| 17 | `EXT-REAL-0008` | `real_00008.jpg` | `543e7e24d9bf...` | **AUTHENTIC** | `real` | `0.2830` | ✅ TRUE | Standard |
| 18 | `EXT-FAKE-0008` | `fake_00008.jpg` | `0b0eb6011aba...` | **MANIPULATED** | `real` | `0.4597` | ❌ FALSE | ⚠️ Low Margin |
| 19 | `EXT-REAL-0009` | `real_00009.jpg` | `82e3d071c934...` | **AUTHENTIC** | `real` | `0.4955` | ✅ TRUE | ⚠️ Low Margin |
| 20 | `EXT-FAKE-0009` | `fake_00009.jpg` | `f44e9f3f163a...` | **MANIPULATED** | `real` | `0.4489` | ❌ FALSE | ⚠️ Low Margin |
| 21 | `EXT-REAL-0010` | `real_00010.jpg` | `0e3922b3c5e8...` | **AUTHENTIC** | `real` | `0.2715` | ✅ TRUE | Standard |
| 22 | `EXT-FAKE-0010` | `fake_00010.jpg` | `3f635cfc8e06...` | **MANIPULATED** | `real` | `0.3715` | ❌ FALSE | Standard |
| 23 | `EXT-REAL-0011` | `real_00011.jpg` | `101ce4d56026...` | **AUTHENTIC** | `real` | `0.1442` | ✅ TRUE | Standard |
| 24 | `EXT-FAKE-0011` | `fake_00011.jpg` | `bfc91377a0af...` | **MANIPULATED** | `fake` | `0.6513` | ✅ TRUE | Standard |
| 25 | `EXT-REAL-0012` | `real_00012.jpg` | `cca0305dad3f...` | **AUTHENTIC** | `real` | `0.4489` | ✅ TRUE | ⚠️ Low Margin |
| 26 | `EXT-FAKE-0012` | `fake_00012.jpg` | `5f8d9496d0ba...` | **MANIPULATED** | `real` | `0.1074` | ❌ FALSE | Standard |
| 27 | `EXT-REAL-0013` | `real_00013.jpg` | `8a18a5612659...` | **AUTHENTIC** | `real` | `0.4876` | ✅ TRUE | ⚠️ Low Margin |
| 28 | `EXT-FAKE-0013` | `fake_00013.jpg` | `a824bf83c3bb...` | **MANIPULATED** | `fake` | `0.7802` | ✅ TRUE | Standard |
| 29 | `EXT-REAL-0014` | `real_00014.jpg` | `10f5013904fb...` | **AUTHENTIC** | `real` | `0.2377` | ✅ TRUE | Standard |
| 30 | `EXT-FAKE-0014` | `fake_00014.jpg` | `3a6f26255a15...` | **MANIPULATED** | `fake` | `0.8562` | ✅ TRUE | Standard |
| 31 | `EXT-REAL-0015` | `real_00015.jpg` | `3f4e8b76b893...` | **AUTHENTIC** | `real` | `0.3948` | ✅ TRUE | Standard |
| 32 | `EXT-FAKE-0015` | `fake_00015.jpg` | `757acbdde25c...` | **MANIPULATED** | `fake` | `0.9712` | ✅ TRUE | Standard |
| 33 | `EXT-REAL-0016` | `real_00016.jpg` | `31740949285a...` | **AUTHENTIC** | `real` | `0.4904` | ✅ TRUE | ⚠️ Low Margin |
| 34 | `EXT-FAKE-0016` | `fake_00016.jpg` | `10c8d5c978cf...` | **MANIPULATED** | `real` | `0.2570` | ❌ FALSE | Standard |
| 35 | `EXT-REAL-0017` | `real_00017.jpg` | `6cf857331b73...` | **AUTHENTIC** | `real` | `0.4740` | ✅ TRUE | ⚠️ Low Margin |
| 36 | `EXT-FAKE-0017` | `fake_00017.jpg` | `27102a2dba83...` | **MANIPULATED** | `real` | `0.3715` | ❌ FALSE | Standard |
| 37 | `EXT-REAL-0018` | `real_00018.jpg` | `a254c71e537a...` | **AUTHENTIC** | `real` | `0.1872` | ✅ TRUE | Standard |
| 38 | `EXT-FAKE-0018` | `fake_00018.jpg` | `8aeb9f76a675...` | **MANIPULATED** | `fake` | `0.5128` | ✅ TRUE | ⚠️ Low Margin |
| 39 | `EXT-REAL-0019` | `real_00019.jpg` | `2277d7e1aed9...` | **AUTHENTIC** | `real` | `0.3246` | ✅ TRUE | Standard |
| 40 | `EXT-FAKE-0019` | `fake_00019.jpg` | `c306d54cdd5b...` | **MANIPULATED** | `real` | `0.3193` | ❌ FALSE | Standard |
| 41 | `EXT-REAL-0020` | `real_00020.jpg` | `72b699243e74...` | **AUTHENTIC** | `real` | `0.0604` | ✅ TRUE | Standard |
| 42 | `EXT-FAKE-0020` | `fake_00020.jpg` | `4ba1dc4586b3...` | **MANIPULATED** | `real` | `0.1283` | ❌ FALSE | Standard |
| 43 | `EXT-REAL-0021` | `real_00021.jpg` | `c2210e7883a8...` | **AUTHENTIC** | `fake` | `0.7138` | ❌ FALSE | Standard |
| 44 | `EXT-FAKE-0021` | `fake_00021.jpg` | `b2323c01f32b...` | **MANIPULATED** | `fake` | `0.9142` | ✅ TRUE | Standard |
| 45 | `EXT-REAL-0022` | `real_00022.jpg` | `d902e6b104e9...` | **AUTHENTIC** | `real` | `0.3758` | ✅ TRUE | Standard |
| 46 | `EXT-FAKE-0022` | `fake_00022.jpg` | `b5b506b181ee...` | **MANIPULATED** | `fake` | `0.9302` | ✅ TRUE | Standard |
| 47 | `EXT-REAL-0023` | `real_00023.jpg` | `00bf2b3d7cbd...` | **AUTHENTIC** | `real` | `0.3616` | ✅ TRUE | Standard |
| 48 | `EXT-FAKE-0023` | `fake_00023.jpg` | `ccb43fa87214...` | **MANIPULATED** | `real` | `0.0597` | ❌ FALSE | Standard |
| 49 | `EXT-REAL-0024` | `real_00024.jpg` | `4e83aa80fb4f...` | **AUTHENTIC** | `real` | `0.2222` | ✅ TRUE | Standard |
| 50 | `EXT-FAKE-0024` | `fake_00024.jpg` | `a3dd1a068962...` | **MANIPULATED** | `fake` | `0.6807` | ✅ TRUE | Standard |
| 51 | `EXT-REAL-0025` | `real_00025.jpg` | `9933f6eeac35...` | **AUTHENTIC** | `real` | `0.2432` | ✅ TRUE | Standard |
| 52 | `EXT-FAKE-0025` | `fake_00025.jpg` | `590cb381ff7f...` | **MANIPULATED** | `real` | `0.3503` | ❌ FALSE | Standard |
| 53 | `EXT-REAL-0026` | `real_00026.jpg` | `248222b1b95a...` | **AUTHENTIC** | `real` | `0.2415` | ✅ TRUE | Standard |
| 54 | `EXT-FAKE-0026` | `fake_00026.jpg` | `ab2f68f6af4c...` | **MANIPULATED** | `fake` | `0.9168` | ✅ TRUE | Standard |
| 55 | `EXT-REAL-0027` | `real_00027.jpg` | `8b62cfbb57ce...` | **AUTHENTIC** | `real` | `0.1384` | ✅ TRUE | Standard |
| 56 | `EXT-FAKE-0027` | `fake_00027.jpg` | `b5f2f51859ff...` | **MANIPULATED** | `real` | `0.0159` | ❌ FALSE | Standard |
| 57 | `EXT-REAL-0028` | `real_00028.jpg` | `7ad40f7edd4f...` | **AUTHENTIC** | `real` | `0.0480` | ✅ TRUE | Standard |
| 58 | `EXT-FAKE-0028` | `fake_00028.jpg` | `b3cfd669045e...` | **MANIPULATED** | `real` | `0.3977` | ❌ FALSE | Standard |
| 59 | `EXT-REAL-0029` | `real_00029.jpg` | `73b3c7e98375...` | **AUTHENTIC** | `real` | `0.2471` | ✅ TRUE | Standard |
| 60 | `EXT-FAKE-0029` | `fake_00029.jpg` | `56ccfb66cf7c...` | **MANIPULATED** | `fake` | `0.5561` | ✅ TRUE | ⚠️ Low Margin |
| 61 | `EXT-REAL-0030` | `real_00030.jpg` | `93d46cf8b968...` | **AUTHENTIC** | `real` | `0.1672` | ✅ TRUE | Standard |
| 62 | `EXT-FAKE-0030` | `fake_00030.jpg` | `12f07bf7519c...` | **MANIPULATED** | `fake` | `0.9502` | ✅ TRUE | Standard |
| 63 | `EXT-REAL-0031` | `real_00031.jpg` | `8d4489629b61...` | **AUTHENTIC** | `real` | `0.3529` | ✅ TRUE | Standard |
| 64 | `EXT-FAKE-0031` | `fake_00031.jpg` | `bd729e06cd83...` | **MANIPULATED** | `real` | `0.2110` | ❌ FALSE | Standard |
| 65 | `EXT-REAL-0032` | `real_00032.jpg` | `40b6859a4053...` | **AUTHENTIC** | `fake` | `0.5252` | ❌ FALSE | ⚠️ Low Margin |
| 66 | `EXT-FAKE-0032` | `fake_00032.jpg` | `b38f6128ecc1...` | **MANIPULATED** | `real` | `0.2663` | ❌ FALSE | Standard |
| 67 | `EXT-REAL-0033` | `real_00033.jpg` | `7dc1ee2876c2...` | **AUTHENTIC** | `real` | `0.3296` | ✅ TRUE | Standard |
| 68 | `EXT-FAKE-0033` | `fake_00033.jpg` | `13df6d627435...` | **MANIPULATED** | `fake` | `0.9866` | ✅ TRUE | Standard |
| 69 | `EXT-REAL-0034` | `real_00034.jpg` | `a7ed6d2da287...` | **AUTHENTIC** | `fake` | `0.7612` | ❌ FALSE | Standard |
| 70 | `EXT-FAKE-0034` | `fake_00034.jpg` | `ae815b60e88d...` | **MANIPULATED** | `fake` | `0.6611` | ✅ TRUE | Standard |
| 71 | `EXT-REAL-0035` | `real_00035.jpg` | `e38f0b2c1851...` | **AUTHENTIC** | `real` | `0.2683` | ✅ TRUE | Standard |
| 72 | `EXT-FAKE-0035` | `fake_00035.jpg` | `c45f4a98ec73...` | **MANIPULATED** | `fake` | `0.6999` | ✅ TRUE | Standard |
| 73 | `EXT-REAL-0036` | `real_00036.jpg` | `49b9ae992914...` | **AUTHENTIC** | `real` | `0.4134` | ✅ TRUE | ⚠️ Low Margin |
| 74 | `EXT-FAKE-0036` | `fake_00036.jpg` | `508688c8a9fc...` | **MANIPULATED** | `fake` | `0.5700` | ✅ TRUE | ⚠️ Low Margin |
| 75 | `EXT-REAL-0037` | `real_00037.jpg` | `2a8b0b64a268...` | **AUTHENTIC** | `real` | `0.2901` | ✅ TRUE | Standard |
| 76 | `EXT-FAKE-0037` | `fake_00037.jpg` | `b01d3716e589...` | **MANIPULATED** | `fake` | `0.9352` | ✅ TRUE | Standard |
| 77 | `EXT-REAL-0038` | `real_00038.jpg` | `da623f61d4e0...` | **AUTHENTIC** | `real` | `0.3732` | ✅ TRUE | Standard |
| 78 | `EXT-FAKE-0038` | `fake_00038.jpg` | `46fb797a549c...` | **MANIPULATED** | `real` | `0.1794` | ❌ FALSE | Standard |
| 79 | `EXT-REAL-0039` | `real_00039.jpg` | `8a87f90635fb...` | **AUTHENTIC** | `real` | `0.1555` | ✅ TRUE | Standard |
| 80 | `EXT-FAKE-0039` | `fake_00039.jpg` | `f6178087fc10...` | **MANIPULATED** | `real` | `0.2263` | ❌ FALSE | Standard |
| 81 | `EXT-REAL-0040` | `real_00040.jpg` | `a5beb6511a19...` | **AUTHENTIC** | `fake` | `0.5412` | ❌ FALSE | ⚠️ Low Margin |
| 82 | `EXT-FAKE-0040` | `fake_00040.jpg` | `81eb8d09d421...` | **MANIPULATED** | `fake` | `0.8472` | ✅ TRUE | Standard |
| 83 | `EXT-REAL-0041` | `real_00041.jpg` | `a5676e66f5cb...` | **AUTHENTIC** | `real` | `0.1409` | ✅ TRUE | Standard |
| 84 | `EXT-FAKE-0041` | `fake_00041.jpg` | `6c762168aa1b...` | **MANIPULATED** | `fake` | `0.5350` | ✅ TRUE | ⚠️ Low Margin |
| 85 | `EXT-REAL-0042` | `real_00042.jpg` | `e57eae374f50...` | **AUTHENTIC** | `real` | `0.3604` | ✅ TRUE | Standard |
| 86 | `EXT-FAKE-0042` | `fake_00042.jpg` | `b5d0623ba441...` | **MANIPULATED** | `fake` | `0.8009` | ✅ TRUE | Standard |
| 87 | `EXT-REAL-0043` | `real_00043.jpg` | `88ff6c55aa96...` | **AUTHENTIC** | `real` | `0.1879` | ✅ TRUE | Standard |
| 88 | `EXT-FAKE-0043` | `fake_00043.jpg` | `905f839dbc12...` | **MANIPULATED** | `real` | `0.2588` | ❌ FALSE | Standard |
| 89 | `EXT-REAL-0044` | `real_00044.jpg` | `bbb5aa441ae5...` | **AUTHENTIC** | `real` | `0.4906` | ✅ TRUE | ⚠️ Low Margin |
| 90 | `EXT-FAKE-0044` | `fake_00044.jpg` | `db28a4ea52e6...` | **MANIPULATED** | `fake` | `0.7010` | ✅ TRUE | Standard |
| 91 | `EXT-REAL-0045` | `real_00045.jpg` | `a302e156ba5e...` | **AUTHENTIC** | `fake` | `0.5147` | ❌ FALSE | ⚠️ Low Margin |
| 92 | `EXT-FAKE-0045` | `fake_00045.jpg` | `ddddb52290ab...` | **MANIPULATED** | `fake` | `0.8243` | ✅ TRUE | Standard |
| 93 | `EXT-REAL-0046` | `real_00046.jpg` | `c9dbbb41c830...` | **AUTHENTIC** | `fake` | `0.5551` | ❌ FALSE | ⚠️ Low Margin |
| 94 | `EXT-FAKE-0046` | `fake_00046.jpg` | `fec388f6892c...` | **MANIPULATED** | `fake` | `0.9865` | ✅ TRUE | Standard |
| 95 | `EXT-REAL-0047` | `real_00047.jpg` | `9227d1167fa3...` | **AUTHENTIC** | `real` | `0.3422` | ✅ TRUE | Standard |
| 96 | `EXT-FAKE-0047` | `fake_00047.jpg` | `3a939a30bfeb...` | **MANIPULATED** | `real` | `0.2762` | ❌ FALSE | Standard |
| 97 | `EXT-REAL-0048` | `real_00048.jpg` | `758165660c80...` | **AUTHENTIC** | `real` | `0.1171` | ✅ TRUE | Standard |
| 98 | `EXT-FAKE-0048` | `fake_00048.jpg` | `0346f1a9d58e...` | **MANIPULATED** | `real` | `0.1805` | ❌ FALSE | Standard |
| 99 | `EXT-REAL-0049` | `real_00049.jpg` | `16e418b45b88...` | **AUTHENTIC** | `real` | `0.4034` | ✅ TRUE | ⚠️ Low Margin |
| 100 | `EXT-FAKE-0049` | `fake_00049.jpg` | `a2a6c7821400...` | **MANIPULATED** | `real` | `0.2690` | ❌ FALSE | Standard |