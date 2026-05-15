const SERVICE_UUID = '12345678-1234-1234-1234-123456789abc';
const CHARACTERISTIC_UUID = 'abcd1234-5678-1234-5678-abcdef123456';

class BLEManager {
  constructor(onDataReceived, onDisconnect) {
    this.device = null;
    this.server = null;
    this.service = null;
    this.characteristic = null;
    this.onDataReceived = onDataReceived; // Callback when data arrives
    this.onDisconnect = onDisconnect; // Callback when disconnected
    this.onDebug = null; // Callback for raw debug info
    this.pollInterval = null; // Store polling interval if needed
  }

  isSupported() {
    return navigator.bluetooth !== undefined;
  }

  async connect() {
    if (!this.isSupported()) {
      throw new Error('Web Bluetooth API is not supported in this browser.');
    }

    try {
      console.log('Requesting Bluetooth Device...');
      this.device = await navigator.bluetooth.requestDevice({
        filters: [{ name: 'PlantCare_ESP32' }],
        optionalServices: [SERVICE_UUID],
      });
      console.log('Device Found:', this.device.name);

      console.log('Connecting to GATT Server...');
      this.device.addEventListener('gattserverdisconnected', this.handleDisconnected.bind(this));
      this.server = await this.device.gatt.connect();
      console.log('Connected to GATT server.');

      // Wait a moment for connection to stabilize (common workaround for Windows/ESP32 stack issues)
      console.log('Waiting 1.5s for connection to stabilize...');
      await new Promise(resolve => setTimeout(resolve, 1500));

      if (!this.device.gatt.connected) {
        throw new Error('GATT Server dropped the connection immediately. Please reset the ESP32 and try again.');
      }

      console.log('Getting Service...', SERVICE_UUID);
      this.service = await this.server.getPrimaryService(SERVICE_UUID);
      console.log('Service Acquired.');

      console.log('Getting Characteristic...', CHARACTERISTIC_UUID);
      this.characteristic = await this.service.getCharacteristic(CHARACTERISTIC_UUID);
      console.log('Characteristic Acquired.');

      console.log('Characteristic Properties:');
      console.log('> Broadcast:', this.characteristic.properties.broadcast);
      console.log('> Read:', this.characteristic.properties.read);
      console.log('> Write w/o response:', this.characteristic.properties.writeWithoutResponse);
      console.log('> Write:', this.characteristic.properties.write);
      console.log('> Notify:', this.characteristic.properties.notify);
      console.log('> Indicate:', this.characteristic.properties.indicate);

      if (this.characteristic.properties.notify) {
        console.log('Starting Notifications...');
        try {
          await this.characteristic.startNotifications();
          this.characteristic.addEventListener('characteristicvaluechanged', this.handleData.bind(this));
          console.log('Notifications enabled successfully.');
        } catch (notifyError) {
          console.warn('Failed to start notifications (missing CCCD?), falling back to polling...', notifyError);
          this.startPolling();
        }
      } else if (this.characteristic.properties.read) {
        console.warn('Notify not supported by property, falling back to polling via readValue()...');
        this.startPolling();
      } else {
        throw new Error('Characteristic does not support Notify or Read.');
      }

      console.log('BLE Connection Established.');
      return true;
    } catch (error) {
      console.error('BLE Connection Failed at step:', error);
      this.resetState();
      throw error;
    }
  }

  startPolling() {
    this.pollInterval = setInterval(async () => {
      if (this.characteristic && this.device && this.device.gatt.connected) {
        try {
          const value = await this.characteristic.readValue();
          this.handleData({ target: { value } });
        } catch (readErr) {
          console.error('Error polling data:', readErr);
        }
      }
    }, 2000); // Poll every 2 seconds
  }

  disconnect() {
    if (this.device && this.device.gatt.connected) {
      this.device.gatt.disconnect();
    } else {
      this.resetState();
    }
  }

  handleDisconnected() {
    console.log('BLE Device Disconnected.');
    this.resetState();
    if (this.onDisconnect) {
      this.onDisconnect();
    }
  }

  handleData(event) {
    const value = event.target.value;
    const decoder = new TextDecoder('utf-8');
    const text = decoder.decode(value).trim();
    
    // Broadcast raw text to debug UI
    if (this.onDebug) {
      this.onDebug(text);
    }

    // Expected format: temp,humidity,soil (e.g., "29.5,70,1800")
    if (text) {
      const parts = text.split(',');
      if (parts.length === 3) {
        const temperature = parseFloat(parts[0]);
        const humidity = parseFloat(parts[1]);
        const soilMoisture = parseFloat(parts[2]);
        
        if (!isNaN(temperature) && !isNaN(humidity) && !isNaN(soilMoisture)) {
          if (this.onDataReceived) {
            this.onDataReceived({ temperature, humidity, soilMoisture });
          }
        }
      } else {
        console.warn('Received BLE data but format was unexpected:', text);
      }
    }
  }

  resetState() {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
    this.device = null;
    this.server = null;
    this.service = null;
    this.characteristic = null;
  }
}

export default BLEManager;
