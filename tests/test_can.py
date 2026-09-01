from yamkit.can import bringup_commands, list_can_interfaces, udev_rules_text


def test_bringup_commands():
    cmds = bringup_commands(["can0"])
    assert cmds == ["sudo ip link set can0 down", "sudo ip link set can0 up type can bitrate 1000000"]


def test_udev_rules():
    txt = udev_rules_text({"ABC": "can_left_l"})
    assert 'ATTRS{serial}=="ABC"' in txt and 'NAME="can_left_l"' in txt


def test_list_fake_sysfs(tmp_path):
    (tmp_path / "eth0").mkdir()
    (tmp_path / "eth0" / "type").write_text("1\n")
    c = tmp_path / "can9"
    (c / "statistics").mkdir(parents=True)
    (c / "type").write_text("280\n")
    (c / "statistics" / "rx_packets").write_text("42\n")
    usb = tmp_path / "usb" / "3-1.2"
    (usb / "3-1.2:1.0").mkdir(parents=True)
    (usb / "serial").write_text("SER123\n")
    (usb / "product").write_text("CANable\n")
    (c / "device").symlink_to(usb / "3-1.2:1.0")
    ifaces = list_can_interfaces(tmp_path)
    assert [i.name for i in ifaces] == ["can9"]
    i = ifaces[0]
    assert i.serial == "SER123" and i.product == "CANable" and i.usb_path == "3-1.2" and i.rx_packets == 42
    assert i.up is False  # `ip` knows nothing about can9
