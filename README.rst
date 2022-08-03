NSniff
======

Ceed is an in vitro experiment that stimulates brain slices and records their activity.

For more information: https://matham.github.io/nsniff/index.html

To install https://matham.github.io/nsniff/installation.html

.. image:: https://img.shields.io/pypi/pyversions/nsniff.svg
    :target: https://pypi.python.org/pypi/nsniff/
    :alt: Supported Python versions

.. image:: https://img.shields.io/pypi/v/nsniff.svg
    :target: https://pypi.python.org/pypi/nsniff/
    :alt: Latest Version on PyPI

.. image:: https://coveralls.io/repos/github/matham/nsniff/badge.svg?branch=main
    :target: https://coveralls.io/github/matham/nsniff?branch=main
    :alt: Coverage status

.. image:: https://github.com/matham/nsniff/workflows/Python%20application/badge.svg
    :target: https://github.com/matham/nsniff/actions
    :alt: Github action status


Protocol file
=============

You can use a CSV file to import a sequence of hardware states and their duration the program can
run through.

The format of the CSV file is something like
``duration,valve_banana,valve_apple,valve_kiwi,valve_orange,mfc_1,mfc_2,key,label``.

There must be a header row that adheres to the labels as follows:

#. ``duration`` is the first row and the duration of this row. That is the state of the hardware will be
   set to the values of the row and then it'll wait for ``duration``.
#. ``value_xxx`` is the next ``n`` columns, which is equal to the number of valves (relays) in the system.
   They are counted from left to right in the GUI which correspond to these ``n`` columns.
   The column label must be prefixed with ``value_`` and can have any suffix. The suffix can
   be used to name the odor value etc.
#. ``value_m`` is the next ``m`` columns, which is equal to the number of MFCs in the system.
   They are counted from left to right in the GUI which correspond to these ``m`` columns.
   The column label must be prefixed with ``mfc_`` and can have any suffix. The suffix can
   be used to name the MFC.
#. ``key`` is a single character that can be used within the GUI to trigger to run through the sequence
   of all the rows that contain this character. Or no character, in which case if no character is entered
   in the GUI, those empty rows will be run through.
#. ``label`` is any string (not containing commas) that is logged

Software installation
=====================

Install the dependencies with

```
sudo apt install python3 python3-dev python3-pip
python3 -m pip install --upgrade pip virtualenv
cd Desktop
python3 -m virtualenv venv
cd venv/
source bin/activate

git clone https://github.com/matham/pymoa-remote.git
pip install -e pymoa-remote/[network]
pip install https://github.com/matham/kivy-trio/archive/master.zip
git clone https://github.com/matham/pymoa.git
pip install -e pymoa/
```

Finally start the server with `KIVY_NO_ARGS=1 pymoa_quart_app --port port --host "ip"`.



auto start
------

sudo nano /lib/systemd/system/pymoa_remote.service

paste::
----

[Unit]
Description=MyMoa-remote server
After=multi-user.target network-online.target systemd-networkd-wait-online.service
Wants=network-online.target systemd-networkd-wait-online.service
Requires=systemd-networkd-wait-online.service


StartLimitIntervalSec=1000
StartLimitBurst=10

[Service]
Restart=on-failure
RestartSec=10s
Environment="PYMOA_REMOTE_PORT=5378"
Environment="PYMOA_REMOTE_HOST=rasppi03wifi.cplab.cornell.edu"
Environment="KIVY_NO_ARGS=1"
Environment="PYMOA_REMOTE_SERVE=/home/pi/Desktop/venv/bin/pymoa_quart_app"
ExecStart=/bin/bash -c '$${PYMOA_REMOTE_SERVE} --port $${PYMOA_REMOTE_PORT} --host $${PYMOA_REMOTE_HOST}'

[Install]
WantedBy=multi-user.target

----

sudo chmod 644 /lib/systemd/system/pymoa_remote.service
sudo systemctl daemon-reload
sudo systemctl enable pymoa_remote.service
sudo systemctl start pymoa_remote

systemctl status pymoa_remote

sudo journalctl -u pymoa_remote
https://serverfault.com/a/413408

Hardware setup
==============

The following describes how to set up a Raspberry Pi to control the Stratuscent sensors, valves, and MFCs
as a remote server that we can connect to and control from another computer on the network.

RPi
---

The RPi may come with a noobs SD card or a blank card. Either way we need to install the latest RPi OS directly
so we can use it headless and connect it to the wifi/ethernet.

Install the recommended OS (currently Raspberry Pi OS 32-bits) following this
`guide <https://www.raspberrypi.org/documentation/installation/installing-images/>`_ using a USB sd card reader.

To configure the RPi so you can connect over ssh and provide it the WiFi credentials you can either:

1. Use the recommended RPI Imager and in its settings before writing the image to the SD card provide
   the WiFi etc.
2. Or we'll provide a file with configuration details after writing to the SD card.
   Once the OS is installed to the card, open the SD card again on a PC and follow this
   `guide <https://www.raspberrypi.org/documentation/configuration/wireless/headless.md>`_. Essentially,
   in the SD root directory of the ``boot`` partition, create a empty file called ``ssh`` this will enable ssh.
   Then create a file called ``wpa_supplicant.conf`` as well with contents::

       country=US
       ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
       update_config=1

       network={
           ssid="network-name"
           psk="password"
       }

   replacing the ssid and psk with the WiFi details.

Then, insert the SD card and power ON the RPi, wait a little bit for it to set up and connect to your router.
Then, log into your router to get the IP of the RPi. Log into the RPi using `ssh pi@ip` with default password
`raspberry` (unless you changed it).

Update the OS using with::

    sudo apt update
    sudo apt dist-upgrade

If you would like to connect directly to RPi using ethernet as it is directly connected to your PC, try following
this `guide <https://bigl.es/friday-fun-connecting-to-your-raspberry-pi/>`_.

Stratuscent sensors
-------------------

https://community.silabs.com/s/article/cp210x-legacy-programming-utilities?language=en_US
https://inegm.medium.com/persistent-names-for-usb-serial-devices-in-linux-dev-ttyusbx-dev-custom-name-fd49b5db9af1
ls /dev/
sudo nano /etc/udev/rules.d/99-usb-serial.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{serial}=="10023B", SYMLINK+="SScent10023B"


Connecting MOD-IO:
-------------------

UART:

sudo raspi-config - Expand filesystem and enable serial on advanced page, exit and reboot.
sudo nano /boot/config.txt - add `dtoverlay=pi3-disable-bt`
sudo nano /boot/cmdline.txt - remove  "console=serial0,115200" or (ttyAMA0)
reboot


I2C:
desolder R15 and R17

https://www.abelectronics.co.uk/kb/article/1/i2c-part-2---enabling-i-c-on-the-raspberry-pi

Enable I2C using `sudo raspi-config`, option 5 and enable it.
Then do
```sh
sudo apt install i2c-tools
// list all devices
sudo i2cdetect -y 1
pip install smbus2
```

It communicates over I2C and boards can be connected sequentially, but they need to be powered independently.
Barrel connector power should be 8V-30V DC.
Blue twin is opto-isolated digital input (same range as power).
Blue triplet is opto-isolated analog output relay (5A/250VAC).
Green is 4 analog inputs (0 - 3.3V).
UEXT is connected to I2C pins of RPi: connect ground, scl to scl and sda to sda.
Connect them in series, they each need power.
Update the address from default 0x58 to e.g. 0x22 using `i2cset -y -f 2 0x58 0xF0 0x22`.


ADC
-----

Enable SPI using `sudo raspi-config`, option 5 and enable it.

Install

sudo apt install python3-pigpio pigpio
pip install pigpio for venv
sudo pigpiod to start daemon

```
git clone https://github.com/ul-gh/PiPyADC.git

```

MFC
---

Keep 120R
leave default switches, full-auto

# In /boot/config.txt:
sudo nano /boot/config.txt
# Add the following, int_pin is set according to the actual welding mode(BCM coding):
dtoverlay=sc16is752-spi1,int_pin=24

After reboot, the driver of SC16IS752 will be loaded into the system
kernel. At this time, you can run ls /dev to see that there will be more
devices as follows:
ttySC0      ttySC1
