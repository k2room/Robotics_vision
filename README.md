# Overview
~~
# Set Jetson Nano
1. Download the Ubuntu 20.04 image from https://github.com/Qengineering/Jetson-Nano-Ubuntu-20-image.
2. Download Balena Etcher from https://etcher.balena.io/.
3. Follow the Write Image to the microSD Card instructions from https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit. You need to allocate at least 64GB of SD card partition using Gparted. Change the value of 'New Size(MiB)' to allocate the enough space.   
   ```
   sudo apt-get install gparted
   ```
4. Check the installations in the terminal.
    ```
    Python3
    >>> Import cv2, torch, torchvision
    >>> cv2.__version__
    >>> torch.__version__
    >>> torchvision.__version__
    ```
# Install OpenCV with CUDA
Building the complete OpenCV package requires more than 4 Gbytes of RAM and the 2 Gbytes of swap space. Once everything is done and the system has rebooted, if you enter the "free -m" command in the terminal and see that the swap space is around 6074, then you have succeeded. Find more information in here: https://qengineering.eu/install-opencv-on-jetson-nano.html.  
1. Check for updates. If you've already done this in a previous post, you can skip it.
   ```
   sudo apt-get update sudo apt-get upgrade
   ```
2. Install the nano text editor and dphys-swapfile.
   ```
   sudo apt-get install nano
   sudo apt-get install dphys-swapfile
   ```
3. Add or uncomment the following lines to set the values of both swap files as shown below.
   ```
   CONF_SWAPSIZE=4096
   CONF_SWAPFACTOR=2
   CONF_MAXSWAP=4096
   ```
4. Open /sbin/dphys-swapfile for editing.
   ```
   sudo nano /sbin/dphys-swapfile
   ```
5. Edit /etc/dphys-swapfile.
   ```
   sudo nano /etc/dphys-swapfile
   ```
6. Reboot the Jetson Nano.
   ```
   sudo reboot
   ```



