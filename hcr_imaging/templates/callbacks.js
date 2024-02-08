// This file holds some Javascript scripts used from Python
// The sections are parsed by the commented callback=whatever lines
// This is not valid Javascript code in isolation; it's just nice for syntax highlighting

/******************************************************************************/

//callback=line_slider
var lines = all_lines.data['lines'];
var y = line.data['y'];
var offset = y.length * slider.value;
for (var r = 0; r < y.length; r++) {y[r] = lines[offset + r];}
line.change.emit()

/******************************************************************************/

//callback=show_zxyk
var data = source.data;
var i = choice.options.indexOf(choice.value);
var zxy = data['zxy'][0][i]; // the data of the whole image (z, x, y, 3)
var xy = data['xy'][0];      // the uint32 data of the selected slice (x, y)
var rgba = new Uint8ClampedArray(xy.buffer); // shape is (x, y, 8)
var offset = Math.round((zslice.value - zslice.start) / zslice.step) * xy.length * 3;

var a = scale / slider.value;
for (var r = 0; r < xy.length; r++) {
    rgba[4 * r + 0] = zxy[offset + 3 * r + 0] * a;
    rgba[4 * r + 1] = zxy[offset + 3 * r + 1] * a;
    rgba[4 * r + 2] = zxy[offset + 3 * r + 2] * a;
}
source.change.emit();

/******************************************************************************/

//callback=show_zxy
var data = source.data;
var i = choice.options.indexOf(choice.value);
var zxy = data['zxy'][0][i]; // the data of the whole image (z, x, y)
var xy = new Float32Array(data['xy'][0].buffer);      // the float32 data of the slice (x, y)
var offset = Math.round((zslice.value - zslice.start) / zslice.step) * xy.length;

var a = scale / slider.value;
for (var r = 0; r < xy.length; r++) {
    xy[r] = Math.min(1, zxy[offset + r] * a);
}
source.change.emit();

/******************************************************************************/

//callback=show_xy
var data = source.data;
var i = choice.options.indexOf(choice.value);
var xy0 = data['xy0'][0][i]; // the selected image (x, y)
var xy = new Float32Array(data['xy'][0].buffer);      // the float32 data of the image (x, y)
var a = scale / slider.value;
for (var r = 0; r < xy.length; r++) {
    xy[r] = Math.min(1, xy0[r] * a);
}
source.change.emit();

/******************************************************************************/

//callback=show_xyk
var data = source.data;
var i = choice.options.indexOf(choice.value);
var xyk = data['xyk'][0][i]; // the selected image (x, y, k)
var xy = data['xy'][0];      // the uint8 data of the image (x, y)
var rgba = new Uint8ClampedArray(xy.buffer); // (x, y, k)

var a = scale / slider.value;
for (var r = 0; r < xy.length; r++) {
    rgba[4 * r + 0] = xyk[3 * r + 0] * a;
    rgba[4 * r + 1] = xyk[3 * r + 1] * a;
    rgba[4 * r + 2] = xyk[3 * r + 2] * a;
}
source.change.emit();

/******************************************************************************/

//callback=rgba

var a = 1 / M.value;
var data = source.data;
var rgba = data['rgba'][0];

var u32 = data['u32'][0];
// u8.shape is (z, x, y, i)
var u8 = data['u8'][0];
var v8 = new Uint8ClampedArray(u32.buffer);
var nc = rgba.length / 4;
var offset =  S.value * u32.length * nc;

for (var r = 0; r < u32.length; r++) {
    v8[4 * r + 0] = 0;
    v8[4 * r + 1] = 0;
    v8[4 * r + 2] = 0;
    v8[4 * r + 3] = 255;
    for (var c = 0; c < nc; c++) {
        v8[4 * r + 0] += u8[offset + nc * r + c] * a * rgba[4 * c + 0];
        v8[4 * r + 1] += u8[offset + nc * r + c] * a * rgba[4 * c + 1];
        v8[4 * r + 2] += u8[offset + nc * r + c] * a * rgba[4 * c + 2];
    }
}
source.change.emit();

/******************************************************************************/

//callback=upload/add

var input = document.createElement('input');
input.setAttribute('type', 'file');
input.setAttribute("multiple", '');
input.onchange = function() {
    var u = source.data.uuid.slice();
    var n = source.data.name.slice();
    var s = source.data.size.slice();
    var t = source.data.time.slice();
    var d = source.data.done.slice();
    var m = source.data.md5.slice();

    for (var f = 0; f < input.files.length; f++) {
        let file = input.files[f];
        let uuid = "";
        for (i = 0; i < 32; i++) {uuid += Math.floor(Math.random() * 16).toString(16);}

        n.push(file.name);
        s.push(file.size / 1.e6);
        t.push(file.lastModified);
        u.push(uuid);
        d.push(false);
        m.push("");

        let xhr = new XMLHttpRequest();
        xhr.open("POST", '/upload');
        let formData = new FormData();
        formData.append("file", file);
        formData.append("upload_id", uuid);
        formData.append("upload_dest", dest);
        xhr.send(formData); // this information will be caught by UploadHandler
    }

    source.data = {'name': n, 'size': s, 'time': t, 'uuid': u, 'done': d, 'md5': m};
    source.change.emit();
}
input.click();

/******************************************************************************/

//callback=upload/clear
source.data = {'uuid': [], 'name': [], 'time': [], 'size': [], 'done': [], 'md5': []};
source.change.emit();
