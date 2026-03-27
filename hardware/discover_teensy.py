import serial.tools.list_ports

def discover_teensys():
    print("--- Teensy Discovery Tool ---")
    ports = serial.tools.list_ports.comports()
    found = False
    
    for port in ports:
        # Teensy boards identify as 'PJRC' or 'Teensy'
        if "Teensy" in port.description or "PJRC" in port.manufacturer:
            found = True
            print(f"Device: {port.device}")
            print(f"  Description: {port.description}")
            print(f"  Serial Number: {port.serial_number}") # <--- COPY THIS
            print("-" * 30)
            
    if not found:
        print("No Teensy boards found. Check your USB cables!")

if __name__ == "__main__":
    discover_teensys()