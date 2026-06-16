import cv2
import time
import sys

from gpiozero import AngularServo, DistanceSensor, OutputDevice, TonalBuzzer
from gpiozero.pins.lgpio import LGPIOFactory
from gpiozero.tones import Tone
from picamera2 import Picamera2
from RPLCD.i2c import CharLCD
from ultralytics import YOLO

# ============================================================
# 설정값
# ============================================================

TRIGGER_PIN = 23
ECHO_PIN = 22          # 현재 네가 성공한 Echo 핀
DETECT_DISTANCE_CM = 60
SCAN_SECONDS = 3
CONF_THRESHOLD = 0.3
RESULT_DISPLAY_SECONDS = 5

LCD_ADDRESS_CANDIDATES = (0x27, 0x3f)
YOLO_MODEL_PATH = "/home/rpi/finalProject/best.pt"

ENV_CONTROL_ENABLED = True
ENV_SENSOR_TYPE = "DHT11"  # "DHT11" 또는 "DHT22"
DHT_DATA_PIN = 4
ENV_LED_PIN = 18
ENV_LED_ACTIVE_HIGH = True
DETECTION_LED_PIN = 17
DETECTION_LED_ACTIVE_HIGH = True
BUZZER_ENABLED = True
BUZZER_PIN = 27
BUZZER_ACTIVE_HIGH = True
BUZZER_VOLUME = 0.5
SERVO_ENABLED = True
SERVO_PIN = 12
SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 180
SERVO_MIN_PULSE_WIDTH = 0.0005
SERVO_MAX_PULSE_WIDTH = 0.0024
SERVO_SETTLE_SECONDS = 0.5
SERVO_DETACH_AFTER_MOVE = True
SERVO_ANGLES = {
    "plastic": 40,
    "paper": 70,
    "metal": 120,
    "glass": 160,
    "unclassified": 0,
}
ENV_CHECK_INTERVAL_SECONDS = 10
ENV_READ_RETRIES = 3
ENV_STATUS_LCD_ENABLED = False
TEMP_HIGH_C = 24.0
TEMP_LOW_C = 23.0
HUMIDITY_HIGH_PERCENT = 70.0
HUMIDITY_LOW_PERCENT = 65.0

try:
    LOCAL_MODEL = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    print("❌ 로컬 YOLO 모델 로드 실패")
    print(e)
    sys.exit(1)

print("🧠 로컬 YOLOv8 모델 사용")
print(f"📦 model={YOLO_MODEL_PATH}")
print(f"🏷️ classes={LOCAL_MODEL.names}")

# ============================================================
# 1. 하드웨어 설정
# ============================================================

try:
    factory = LGPIOFactory()
    print("✅ LGPIOFactory 초기화 성공")
except Exception as e:
    print("⚠️ LGPIOFactory 기본 초기화 실패")
    print(e)
    print("⚠️ chip=0으로 재시도")
    factory = LGPIOFactory(chip=0)

# HC-SR04 초음파 센서
ultrasonic = DistanceSensor(
    echo=ECHO_PIN,
    trigger=TRIGGER_PIN,
    max_distance=2.0,
    pin_factory=factory
)

# I2C LCD
lcd = None

for lcd_address in LCD_ADDRESS_CANDIDATES:
    try:
        lcd = CharLCD(
            "PCF8574",
            lcd_address,
            port=1,
            cols=16,
            rows=2
        )
        print(f"📟 I2C LCD 초기화 성공: 0x{lcd_address:02x}")
        break

    except Exception as e:
        print(f"⚠️ LCD 초기화 실패: 0x{lcd_address:02x}")
        print(e)

if not lcd:
    print("⚠️ LCD 없이 계속 실행합니다")


env_control_active = False
dht_sensor = None
env_led = None
env_led_is_on = False
last_env_check = 0.0
detection_led = None
detection_led_is_on = False
camera_is_running = False
buzzer = None
sort_servo = None


def lcd_print(line1="", line2=""):
    if not lcd:
        print(f"⚠️ LCD 출력 생략: {line1} / {line2}")
        return

    try:
        first_line = str(line1)[:16]
        second_line = str(line2)[:16]

        lcd.clear()
        lcd.write_string(first_line)

        if second_line:
            lcd.write_string("\n" + second_line)

    except Exception as e:
        print("⚠️ LCD 출력 실패:", e)


def get_dht_board_pin(board_module):
    pin_name = f"D{DHT_DATA_PIN}"

    if not hasattr(board_module, pin_name):
        raise ValueError(f"board.{pin_name} 핀을 찾을 수 없습니다")

    return getattr(board_module, pin_name)


class LgpioDHTSensor:
    def __init__(self, sensor_type, data_pin):
        import lgpio

        self.lgpio = lgpio
        self.sensor_type = sensor_type
        self.data_pin = data_pin
        self.chip = lgpio.gpiochip_open(0)

    def exit(self):
        try:
            self.lgpio.gpio_free(self.chip, self.data_pin)
        except Exception:
            pass

        try:
            self.lgpio.gpiochip_close(self.chip)
        except Exception:
            pass

    def probe_data_pin(self):
        lgpio = self.lgpio

        try:
            lgpio.gpio_free(self.chip, self.data_pin)
        except Exception:
            pass

        lgpio.gpio_claim_input(self.chip, self.data_pin, lgpio.SET_PULL_UP)
        time.sleep(0.02)
        level = lgpio.gpio_read(self.chip, self.data_pin)

        try:
            lgpio.gpio_free(self.chip, self.data_pin)
        except Exception:
            pass

        return level

    def read(self):
        last_error = None

        for _ in range(ENV_READ_RETRIES):
            try:
                return self._read_once()
            except RuntimeError as e:
                last_error = e
                time.sleep(0.25)

        raise RuntimeError(last_error)

    def _read_once(self):
        lgpio = self.lgpio
        events = []

        def edge_callback(chip, gpio, level, tick):
            if level in (0, 1):
                events.append((level, tick))

        callback = None

        try:
            try:
                lgpio.gpio_free(self.chip, self.data_pin)
            except Exception:
                pass

            lgpio.gpio_claim_output(
                self.chip,
                self.data_pin,
                1,
                lgpio.SET_OPEN_DRAIN
            )
            time.sleep(0.05)

            lgpio.gpio_write(self.chip, self.data_pin, 0)
            time.sleep(0.02)

            try:
                lgpio.gpio_free(self.chip, self.data_pin)
            except Exception:
                pass

            lgpio.gpio_claim_alert(
                self.chip,
                self.data_pin,
                lgpio.BOTH_EDGES,
                lgpio.SET_PULL_UP
            )

            callback = lgpio.callback(
                self.chip,
                self.data_pin,
                lgpio.BOTH_EDGES,
                edge_callback
            )

            time.sleep(0.12)

        finally:
            if callback:
                callback.cancel()

        data_pulses = self._extract_data_high_pulses(events)
        return self._decode_pulses(data_pulses)

    def _extract_data_high_pulses(self, events):
        high_pulses_us = []
        rise_tick = None

        for level, tick in events:
            if level == 1:
                rise_tick = tick
            elif level == 0 and rise_tick is not None:
                high_us = (tick - rise_tick) / 1000.0
                rise_tick = None

                if 10 <= high_us <= 120:
                    high_pulses_us.append(high_us)

        if len(high_pulses_us) < 40:
            raise RuntimeError(
                f"Only {len(high_pulses_us)} data pulses captured"
            )

        return high_pulses_us[-40:]

    def _decode_pulses(self, data_pulses):
        bits = [1 if pulse_us > 50 else 0 for pulse_us in data_pulses]
        data = []

        for byte_index in range(5):
            value = 0

            for bit_index in range(8):
                value = (value << 1) | bits[byte_index * 8 + bit_index]

            data.append(value)

        checksum = sum(data[:4]) & 0xFF

        if checksum != data[4]:
            raise RuntimeError(
                f"Checksum mismatch: got {data[4]}, expected {checksum}"
            )

        if self.sensor_type == "DHT11":
            humidity_percent = data[0] + data[1] / 10.0
            temperature_c = data[2] + data[3] / 10.0
        else:
            raw_humidity = (data[0] << 8) | data[1]
            raw_temperature = (data[2] << 8) | data[3]

            humidity_percent = raw_humidity / 10.0

            if raw_temperature & 0x8000:
                raw_temperature &= 0x7FFF
                temperature_c = -(raw_temperature / 10.0)
            else:
                temperature_c = raw_temperature / 10.0

        return float(temperature_c), float(humidity_percent)


def get_env_sensor_type():
    sensor_type = ENV_SENSOR_TYPE.upper()

    if sensor_type not in ("DHT11", "DHT22"):
        raise ValueError("ENV_SENSOR_TYPE은 DHT11 또는 DHT22만 지원합니다")

    return sensor_type


def create_adafruit_dht_sensor(sensor_type):
    import adafruit_dht
    import board

    if sensor_type == "DHT11":
        sensor_class = adafruit_dht.DHT11
    else:
        sensor_class = adafruit_dht.DHT22

    return sensor_class(get_dht_board_pin(board))


def create_lgpio_dht_sensor(sensor_type):
    sensor = LgpioDHTSensor(sensor_type, DHT_DATA_PIN)

    print("🌡️ lgpio DHT fallback 초기화 성공")

    try:
        level = sensor.probe_data_pin()
        print(f"🌡️ DHT DATA 초기 레벨: {'HIGH' if level else 'LOW'}")
    except Exception as e:
        print("⚠️ DHT DATA 초기 레벨 확인 실패:", e)

    return sensor


def create_dht_sensor(sensor_type):
    try:
        sensor = create_adafruit_dht_sensor(sensor_type)
        print("🌡️ adafruit_dht 센서 초기화 성공")
        return sensor, "adafruit_dht"
    except ImportError as e:
        print("⚠️ adafruit_dht 없음: lgpio fallback 사용")
        print(e)
    except Exception as e:
        print("⚠️ adafruit_dht 초기화 실패: lgpio fallback 사용")
        print(e)

    sensor = create_lgpio_dht_sensor(sensor_type)
    return sensor, "lgpio"


def init_environment_control():
    global env_control_active
    global dht_sensor
    global env_led

    if not ENV_CONTROL_ENABLED:
        print("🌡️ 환경 제어 비활성화")
        return

    try:
        sensor_type = get_env_sensor_type()
        dht_sensor, sensor_backend = create_dht_sensor(sensor_type)
        env_led = OutputDevice(
            ENV_LED_PIN,
            active_high=ENV_LED_ACTIVE_HIGH,
            initial_value=False,
            pin_factory=factory
        )
        env_control_active = True

        print(
            f"🌡️ 환경 제어 활성화: {sensor_type} GPIO{DHT_DATA_PIN}, "
            f"backend={sensor_backend}, "
            f"ENV LED GPIO{ENV_LED_PIN}"
        )

    except Exception as e:
        env_control_active = False
        dht_sensor = None

        if env_led:
            try:
                env_led.off()
            except Exception:
                pass

        env_led = None
        print("⚠️ 환경 제어 초기화 실패: 쓰레기 분류 기능만 계속 실행")
        print(e)


def read_environment():
    if not env_control_active or not dht_sensor:
        return None

    try:
        if hasattr(dht_sensor, "read"):
            temperature_c, humidity_percent = dht_sensor.read()
        else:
            temperature_c = dht_sensor.temperature
            humidity_percent = dht_sensor.humidity

    except RuntimeError as e:
        print("⚠️ 온습도 센서 읽기 실패:", e)
        return None
    except Exception as e:
        print("⚠️ 온습도 센서 오류:", e)
        return None

    if temperature_c is None or humidity_percent is None:
        print("⚠️ 온습도 센서 값 없음")
        return None

    return float(temperature_c), float(humidity_percent)


def set_env_led_state(should_turn_on):
    global env_led_is_on

    if not env_led or should_turn_on == env_led_is_on:
        return

    if should_turn_on:
        env_led.on()
        env_led_is_on = True
        print(f"💡 환경 LED ON: GPIO{ENV_LED_PIN}")
    else:
        env_led.off()
        env_led_is_on = False
        print(f"💡 환경 LED OFF: GPIO{ENV_LED_PIN}")


def update_env_led_for_environment(temperature_c, humidity_percent):
    if env_led_is_on:
        should_turn_on = not (
            temperature_c < TEMP_LOW_C
            and humidity_percent < HUMIDITY_LOW_PERCENT
        )
    else:
        should_turn_on = (
            temperature_c >= TEMP_HIGH_C
            or humidity_percent >= HUMIDITY_HIGH_PERCENT
        )

    set_env_led_state(should_turn_on)


def maybe_check_environment():
    global last_env_check

    if not env_control_active:
        return

    now = time.time()

    if now - last_env_check < ENV_CHECK_INTERVAL_SECONDS:
        return

    last_env_check = now
    reading = read_environment()

    if not reading:
        return

    temperature_c, humidity_percent = reading

    update_env_led_for_environment(temperature_c, humidity_percent)

    print(
        f"🌡️ 온습도 센서 체크: "
        f"온도={temperature_c:.1f}C, "
        f"습도={humidity_percent:.1f}%, "
        f"환경LED={'ON' if env_led_is_on else 'OFF'}"
    )

    if ENV_STATUS_LCD_ENABLED:
        lcd_print(
            f"Temp {temperature_c:.1f}C",
            f"H {humidity_percent:.1f}% LED {'ON' if env_led_is_on else 'OFF'}"
        )


def stop_environment_control():
    global env_led_is_on

    if env_led:
        try:
            env_led.off()
            env_led_is_on = False
        except Exception:
            pass

    if dht_sensor:
        try:
            dht_sensor.exit()
        except Exception:
            pass


init_environment_control()


try:
    detection_led = OutputDevice(
        DETECTION_LED_PIN,
        active_high=DETECTION_LED_ACTIVE_HIGH,
        initial_value=False,
        pin_factory=factory
    )
    print(f"💡 감지 LED 준비: GPIO{DETECTION_LED_PIN}")
except Exception as e:
    detection_led = None
    print("⚠️ 감지 LED 초기화 실패")
    print(e)


def init_buzzer():
    if not BUZZER_ENABLED:
        print("🔇 버저 비활성화")
        return None

    try:
        if not BUZZER_ACTIVE_HIGH:
            print("⚠️ TonalBuzzer는 active_high 설정을 직접 사용하지 않습니다")

        buzzer_device = TonalBuzzer(
            BUZZER_PIN,
            mid_tone=Tone("A4"),
            octaves=2,
            pin_factory=factory
        )
        print(f"🔊 버저 준비: GPIO{BUZZER_PIN}")
        return buzzer_device

    except Exception as e:
        print("⚠️ 버저 초기화 실패: 소리 없이 계속 실행")
        print(e)
        return None


def play_tone_sequence(sequence):
    if not buzzer:
        return

    try:
        for tone_name, duration_seconds, pause_seconds in sequence:
            buzzer.play(tone_name)
            buzzer.pwm_device.value = BUZZER_VOLUME
            time.sleep(duration_seconds)
            buzzer.stop()
            time.sleep(pause_seconds)

    except Exception as e:
        print("⚠️ 버저 재생 실패:", e)
        try:
            buzzer.stop()
        except Exception:
            pass


def play_buzzer_pattern(display_name):
    comparable_name = str(display_name).lower().replace("_", " ").replace("-", " ")

    if "plastic" in comparable_name:
        sequence = (
            ("C5", 0.10, 0.04),
            ("E5", 0.10, 0.04),
            ("G5", 0.14, 0.05),
        )
    elif "paper" in comparable_name or "cardboard" in comparable_name:
        sequence = (
            ("C4", 0.18, 0.06),
            ("D4", 0.18, 0.08),
            ("C4", 0.18, 0.04),
        )
    elif "can" in comparable_name or "metal" in comparable_name:
        sequence = (
            ("A5", 0.08, 0.05),
            ("A5", 0.08, 0.05),
        )
    elif "glass" in comparable_name:
        sequence = (
            ("G5", 0.20, 0.06),
            ("D5", 0.12, 0.05),
        )
    else:
        sequence = (
            ("A4", 0.18, 0.05),
        )

    play_tone_sequence(sequence)


def play_buzzer_failure():
    sequence = (
        ("C4", 0.10, 0.05),
        ("C4", 0.10, 0.05),
    )
    play_tone_sequence(sequence)


def stop_buzzer():
    if not buzzer:
        return

    try:
        buzzer.stop()
    except Exception:
        pass


def init_sort_servo():
    if not SERVO_ENABLED:
        print("↔️ 분류 서보 비활성화")
        return None

    try:
        servo_device = AngularServo(
            SERVO_PIN,
            initial_angle=SERVO_ANGLES["unclassified"],
            min_angle=SERVO_MIN_ANGLE,
            max_angle=SERVO_MAX_ANGLE,
            min_pulse_width=SERVO_MIN_PULSE_WIDTH,
            max_pulse_width=SERVO_MAX_PULSE_WIDTH,
            pin_factory=factory
        )
        print(
            f"↔️ SG90 분류 서보 준비: GPIO{SERVO_PIN}, "
            f"initial={SERVO_ANGLES['unclassified']}도"
        )
        return servo_device

    except Exception as e:
        print("⚠️ 분류 서보 초기화 실패: 서보 없이 계속 실행")
        print(e)
        return None


def stop_sort_servo():
    if not sort_servo:
        return

    try:
        sort_servo.detach()
    except Exception:
        pass

    try:
        sort_servo.close()
    except Exception:
        pass


buzzer = init_buzzer()
sort_servo = init_sort_servo()


def set_detection_led_state(should_turn_on):
    global detection_led_is_on

    if not detection_led or should_turn_on == detection_led_is_on:
        return

    if should_turn_on:
        detection_led.on()
        detection_led_is_on = True
        print(f"💡 감지 LED ON: GPIO{DETECTION_LED_PIN}")
    else:
        detection_led.off()
        detection_led_is_on = False
        print(f"💡 감지 LED OFF: GPIO{DETECTION_LED_PIN}")


# ============================================================
# 3. Picamera2 설정
# ============================================================

print("📷 Picamera2 시작 중...")

picam2 = Picamera2()

picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (640, 480)}
    )
)

print("✅ 카메라 설정 완료")


def start_camera():
    global camera_is_running

    if camera_is_running:
        return

    print("📷 카메라 ON")
    picam2.start()
    camera_is_running = True
    time.sleep(2)


def stop_camera():
    global camera_is_running

    if not camera_is_running:
        return

    print("📷 카메라 OFF")

    try:
        picam2.stop()
    except Exception:
        pass

    camera_is_running = False

# ============================================================
# 4. 초기 화면
# ============================================================

lcd_print("Waiting")

print("🚮 쓰레기 자동 분류 시스템 준비 완료")
print(f"📏 감지 거리: {DETECT_DISTANCE_CM}cm 이내")
print(f"📌 TRIG=GPIO{TRIGGER_PIN}, ECHO=GPIO{ECHO_PIN}")

# ============================================================
# 종료 함수
# ============================================================

def stop_system():
    print("\n🔴 시스템 종료")

    lcd_print("System Offline", "")
    stop_environment_control()
    stop_camera()
    stop_buzzer()
    stop_sort_servo()

    set_detection_led_state(False)

    cv2.destroyAllWindows()
    sys.exit(0)


# ============================================================
# 프레임 변환 함수
# ============================================================

def convert_frame_to_bgr(frame):
    # Picamera2가 4채널 XBGR/RGBA 계열로 줄 때 대응
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # 3채널이면 RGB -> BGR
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame


def apply_color_channel_fix(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def predict_with_local_yolo(frame_bgr):
    try:
        results = LOCAL_MODEL.predict(
            frame_bgr,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        if not results:
            return []

        result = results[0]
        names = getattr(result, "names", LOCAL_MODEL.names)
        predictions = []

        if result.boxes is None:
            return []

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = names.get(cls_id, f"class_{cls_id}")

            if conf >= CONF_THRESHOLD:
                predictions.append({
                    "confidence": conf,
                    "class_id": cls_id,
                    "class": class_name,
                })

        return predictions

    except Exception as e:
        print("⚠️ 로컬 YOLO 추론 실패:", e)

    return []


def normalize_display_name(class_name):
    normalized_name = str(class_name).strip()
    comparable_name = normalized_name.lower().replace("_", " ").replace("-", " ")

    if comparable_name in (
        "plastic bag",
        "plastic cup",
        "plastic bottle",
        "container for household chemicals",
    ):
        return "plastic"

    if comparable_name == "cardboard":
        return "Paper"

    if comparable_name == "organic":
        return "unclassified"

    return normalized_name


def get_lcd_display_name(display_name):
    comparable_name = str(display_name).strip().lower()
    comparable_name = comparable_name.replace("_", " ").replace("-", " ")

    if "plastic" in comparable_name:
        return "Plastic"

    if "household chemicals" in comparable_name:
        return "Plastic"

    if "paper" in comparable_name or "cardboard" in comparable_name:
        return "Paper"

    if "aluminum" in comparable_name or "can" in comparable_name or "tin" in comparable_name:
        return "Can"

    if "metal" in comparable_name:
        return "Metal"

    if "glass" in comparable_name or "bottle" in comparable_name:
        return "Glass"

    if comparable_name == "unclassified":
        return "Unknown"

    return str(display_name).strip()[:8]


def get_servo_category(display_name):
    comparable_name = str(display_name).strip().lower()
    comparable_name = comparable_name.replace("_", " ").replace("-", " ")

    if not comparable_name or comparable_name == "unclassified":
        return "unclassified"

    if "plastic" in comparable_name:
        return "plastic"

    if "household chemicals" in comparable_name:
        return "plastic"

    if (
        "paper" in comparable_name
        or "cardboard" in comparable_name
        or "book" in comparable_name
    ):
        return "paper"

    if (
        "can" in comparable_name
        or "metal" in comparable_name
        or "tin" in comparable_name
    ):
        return "metal"

    if "glass" in comparable_name or "bottle" in comparable_name:
        return "glass"

    return "unclassified"


def move_sort_servo(display_name):
    category = get_servo_category(display_name)
    angle = SERVO_ANGLES.get(category, SERVO_ANGLES["unclassified"])

    if not sort_servo:
        print(f"⚠️ 서보 이동 생략: category={category}, angle={angle}")
        return category

    try:
        sort_servo.angle = angle
        print(f"↔️ 서보 이동: category={category}, angle={angle}")
        time.sleep(SERVO_SETTLE_SECONDS)

        if SERVO_DETACH_AFTER_MOVE:
            sort_servo.detach()
            print("↔️ 서보 PWM detach")

    except Exception as e:
        print("⚠️ 서보 이동 실패:", e)

    return category


# ============================================================
# 메인 루프
# ============================================================

try:
    while True:
        distance_cm = ultrasonic.distance * 100

        # 필요하면 거리 확인용
        print(f"거리: {distance_cm:.1f} cm")

        if distance_cm < DETECT_DISTANCE_CM:

            set_detection_led_state(True)
            start_camera()

            print(f"\n🚨 물체 감지 ({distance_cm:.1f} cm)")

            lcd_print("Scanning")

            scan_start = time.time()
            detected_name = None
            max_conf = 0.0
            best_frame = None

            # 한 번 감지되면 3초 동안 무조건 스캔
            while time.time() - scan_start < SCAN_SECONDS:

                print("📸 scanning...")

                frame = picam2.capture_array()
                frame_bgr = convert_frame_to_bgr(frame)
                frame_fixed = apply_color_channel_fix(frame_bgr)

                predictions = predict_with_local_yolo(frame_fixed)

                for prediction in predictions:
                    conf = float(prediction.get("confidence", 0.0))
                    cls_id = prediction.get("class_id", "unknown")
                    raw_name = prediction.get("class") or f"class_{cls_id}"
                    name = normalize_display_name(raw_name)

                    print(
                        f"DEBUG YOLO detect: id={cls_id}, "
                        f"name={raw_name} -> {name}, "
                        f"conf={conf:.2f}"
                    )

                    if name and conf > max_conf:
                        max_conf = conf
                        detected_name = str(name)
                        best_frame = frame_fixed.copy()

                if detected_name:
                    break

                time.sleep(0.2)

            # ====================================================
            # 분류 성공
            # ====================================================

            if detected_name:

                print(
                    f"♻️ {detected_name} "
                    f"({max_conf * 100:.1f}%)"
                )

                lcd_print(
                    get_lcd_display_name(detected_name)
                )

                # 로그 저장
                log_time = time.strftime("%Y-%m-%d %H:%M:%S")

                with open(
                    "garbage_logs_class6_test.txt",
                    "a",
                    encoding="utf-8"
                ) as f:
                    f.write(
                        f"[{log_time}] "
                        f"{detected_name} "
                        f"({max_conf:.2f})\n"
                    )

                print("📝 로그 저장 완료")

                # 이미지 저장
                filename = (
                    "color_fixed_detected_"
                    + time.strftime("%Y%m%d_%H%M%S")
                    + ".jpg"
                )

                if best_frame is not None:
                    cv2.imwrite(filename, best_frame)
                    print(f"💾 이미지 저장: {filename}")

                move_sort_servo(detected_name)
                play_buzzer_pattern(detected_name)

                # 중복 인식 방지
                time.sleep(RESULT_DISPLAY_SECONDS)

            else:
                print("⚠️ 분류 실패: YOLO 감지 결과 없음")
                lcd_print("No Detection", "Try Again")
                move_sort_servo("unclassified")
                play_buzzer_failure()
                time.sleep(1)

            print("🟢 대기 상태 복귀")

            lcd_print("Waiting")
            stop_camera()

        else:
            set_detection_led_state(False)
            stop_camera()
            maybe_check_environment()

        time.sleep(0.3)

except KeyboardInterrupt:
    stop_system()
