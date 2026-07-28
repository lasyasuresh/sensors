import time, glob, smbus2, numpy as np
from collections import deque

# ================= DHT11 via kernel IIO driver =================
def _find_iio():
    for d in glob.glob('/sys/bus/iio/devices/iio:device*'):
        if glob.glob(d + '/in_temp_input'):
            return d
    return None

_IIO = _find_iio()
_dht_last = 0.0
_dht_cache = (None, None)

def _read_iio(path):
    with open(path) as f:
        return int(f.read().strip()) / 1000.0

def read_dht():
    """(temp_c, humidity). Cached ~2.5s — DHT11 hardware limit."""
    global _dht_last, _dht_cache, _IIO
    if time.time() - _dht_last < 2.5:
        return _dht_cache
    _dht_last = time.time()
    if _IIO is None:
        _IIO = _find_iio()
        if _IIO is None:
            return _dht_cache
    try:
        t = _read_iio(_IIO + '/in_temp_input')
        h = _read_iio(_IIO + '/in_humidityrelative_input')
        if -10 < t < 60 and 0 <= h <= 100:
            _dht_cache = (t, h)
    except OSError:
        pass
    return _dht_cache


# ================= MAX30100 =================
class MAX30100:
    ADDR = 0x57
    R_FIFO_WR   = 0x02
    R_FIFO_RD   = 0x04
    R_FIFO_DATA = 0x05
    R_MODE      = 0x06
    R_SPO2      = 0x07
    R_LED       = 0x09
    R_PART_ID   = 0xFF

    def __init__(self, bus=1, led_current=0x77):
        self.b = smbus2.SMBus(bus)
        pid = self.b.read_byte_data(self.ADDR, self.R_PART_ID)
        if pid != 0x11:
            raise RuntimeError(f"Not a MAX30100 (part id {hex(pid)})")
        self._w(self.R_MODE, 0x40)
        time.sleep(0.1)
        self._w(self.R_MODE, 0x03)
        self._w(self.R_SPO2, 0x40 | (0x01 << 2) | 0x03)
        self._w(self.R_LED, led_current)
        self._w(self.R_FIFO_WR, 0)
        self._w(self.R_FIFO_RD, 0)

    def _w(self, r, v):
        self.b.write_byte_data(self.ADDR, r, v)

    def read_samples(self):
        """Drain FIFO. Returns (ir, red). MAX30100 order: IR then RED."""
        wr = self.b.read_byte_data(self.ADDR, self.R_FIFO_WR) & 0x0F
        rd = self.b.read_byte_data(self.ADDR, self.R_FIFO_RD) & 0x0F
        n = (wr - rd) & 0x0F
        if n == 0:
            return [], []
        ir, red = [], []
        for _ in range(n):
            d = self.b.read_i2c_block_data(self.ADDR, self.R_FIFO_DATA, 4)
            ir.append((d[0] << 8) | d[1])
            red.append((d[2] << 8) | d[3])
        return ir, red


# ================= Signal processing =================
FS = 100
FINGER_THRESHOLD = 15000        # your no-finger ~600, with-finger ~30000-48000

def _bandpass(x, fs=FS):
    """Remove DC baseline without edge artifacts. Trims fs//2 from each end."""
    x = np.asarray(x, float)
    w = fs // 2                     # 0.5 s baseline window
    if len(x) < w * 3:
        return np.zeros(0)
    pad = w // 2
    xp = np.pad(x, pad, mode='edge')
    baseline = np.convolve(xp, np.ones(w) / w, mode='same')[pad:pad + len(x)]
    ac = x - baseline
    sp = np.pad(ac, 2, mode='edge')
    ac = np.convolve(sp, np.ones(5) / 5, mode='same')[2:2 + len(ac)]
    return ac[w:-w]                 # drop remaining edge influence

def finger_present(ir):
    return bool(len(ir) > 0 and np.mean(ir) > FINGER_THRESHOLD)


_hr_history = deque(maxlen=7)
_spo2_history = deque(maxlen=7)

def reset_hr_history():
    _hr_history.clear()
    _spo2_history.clear()

def compute_hr(ir, fs=FS):
    """Autocorrelation HR — robust to dicrotic notches and dropped beats."""
    if len(ir) < fs * 3:
        return None
    if np.mean(ir) < FINGER_THRESHOLD:
        _hr_history.clear()
        return None

    s = _bandpass(ir, fs)
    if len(s) < fs * 2:
        return None
    s = s - s.mean()
    if np.std(s) < 1.0:
        return None

    ac = np.correlate(s, s, mode='full')[len(s) - 1:]
    ac = ac / (ac[0] + 1e-9)

    lo = int(fs * 60 / 200)      # 200 bpm
    hi = min(int(fs * 60 / 40), len(ac) - 1)   # 40 bpm
    if hi <= lo:
        return None

    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < 0.35:           # not periodic enough to trust
        return None

    if 0 < lag < len(ac) - 1:    # sub-sample precision
        y0, y1, y2 = ac[lag - 1], ac[lag], ac[lag + 1]
        d = y0 - 2 * y1 + y2
        if abs(d) > 1e-9:
            lag = lag + 0.5 * (y0 - y2) / d

    hr = 60.0 * fs / lag
    if not (40 < hr < 200):
        return None

    _hr_history.append(hr)
    return float(np.median(_hr_history))


def compute_spo2(ir, red):
    """Ratio-of-ratios. Generic curve — uncalibrated, trends only."""
    if len(ir) < FS * 2:
        return None
    ir_a, red_a = np.asarray(ir, float), np.asarray(red, float)
    if ir_a.mean() < FINGER_THRESHOLD:
        return None
    ir_dc, red_dc = ir_a.mean(), red_a.mean()
    if ir_dc <= 0 or red_dc <= 0:
        return None
    ir_f, red_f = _bandpass(ir_a), _bandpass(red_a)
    if len(ir_f) < FS or len(red_f) < FS:
        return None
    ir_ac, red_ac = np.std(ir_f), np.std(red_f)
    if ir_ac <= 0:
        return None
    R = (red_ac / red_dc) / (ir_ac / ir_dc)
    val = float(np.clip(110.0 - 25.0 * R, 70, 100))
    _spo2_history.append(val)
    return float(np.median(_spo2_history))

    # ================= Full diagnostic analysis =================
def analyze(ir, red, fs=FS):
    """One PPG window -> everything we want to store.

    Returns dict with hr, spo2, finger, ir_mean, ir_ac, signal_quality.
    signal_quality is the normalised autocorrelation peak (0-1); values
    below ~0.35 mean the waveform isn't periodic enough to trust.
    """
    out = {"hr": None, "spo2": None, "finger": False,
           "ir_mean": None, "ir_ac": None, "signal_quality": None}

    if len(ir) < fs * 3:
        return out

    ir_a = np.asarray(ir, float)
    out["ir_mean"] = float(ir_a.mean())
    out["finger"] = bool(out["ir_mean"] > FINGER_THRESHOLD)

    if not out["finger"]:
        _hr_history.clear()
        _spo2_history.clear()
        return out

    s = _bandpass(ir_a, fs)
    if len(s) < fs * 2:
        return out
    s = s - s.mean()
    out["ir_ac"] = float(np.ptp(s))

    if np.std(s) < 1.0:
        return out

    ac = np.correlate(s, s, mode='full')[len(s) - 1:]
    ac = ac / (ac[0] + 1e-9)

    lo = int(fs * 60 / 200)
    hi = min(int(fs * 60 / 40), len(ac) - 1)
    if hi <= lo:
        return out

    lag = lo + int(np.argmax(ac[lo:hi]))
    out["signal_quality"] = float(ac[lag])

    # SpO2 uses its own gating, independent of HR confidence
    out["spo2"] = compute_spo2(ir, red)

    if ac[lag] < 0.35:
        return out

    if 0 < lag < len(ac) - 1:
        y0, y1, y2 = ac[lag - 1], ac[lag], ac[lag + 1]
        d = y0 - 2 * y1 + y2
        if abs(d) > 1e-9:
            lag = lag + 0.5 * (y0 - y2) / d

    hr = 60.0 * fs / lag
    if 40 < hr < 200:
        _hr_history.append(hr)
        out["hr"] = float(np.median(_hr_history))

    return out