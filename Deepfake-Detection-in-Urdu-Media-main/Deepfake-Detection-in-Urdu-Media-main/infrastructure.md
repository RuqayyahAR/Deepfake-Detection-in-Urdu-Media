Markdown
# Infrastructure Layout: Host Development via Remote-SSH

This project utilizes a split-system development architecture. Heavy computational models, media dependencies (`FFmpeg`), and data layers (`PostgreSQL`) run natively inside a background **Kali Linux VM**, while the user interface and code drafting are managed via **Visual Studio Code** on the host machine via a secure SSH tunnel.

---

##  Architecture Overview

By offloading the execution environment to a dedicated back-end VM, the local machine saves overhead processing power by disabling the Linux graphical subsystem (headless execution).

```text
+-----------------------+                    +-----------------------+
|  Host Machine         |                    |  Kali Linux VM        |
|  (Windows/macOS UI)   |                    |  (Headless Back-end)  |
|                       |   Secure SSH       |                       |
|  +-----------------+  |   Connection       |  +-----------------+  |
|  |   VS Code UI    |======================>|  | VS Code Server  |  |
|  +-----------------+  |   (Port 22)        |  +-----------------+  |
|                       |                    |  | Python Pipeline |  |
|                       |                    |  | FFmpeg / Models |  |
+-----------------------+                    +-----------------------+
Key Engineering Benefits:
Resource Preservation: Bypassing the Kali Desktop Environment (Xfce/GNOME) reclaims up to 1.5 GB of RAM, allocating it completely to the processing pipelines of Whisper and PyTorch.

Unified Environment: Eliminates the classic cross-platform path bug ("it works on my machine") by standardizing operations inside a POSIX-compliant Linux file system.

Decoupled Dependencies: Native binary system packages like ffmpeg run directly inside the back-end environment, ignoring local Windows system variable limitations.

## Step-by-Step Environment Configuration
Step 1: Provision the Back-end SSH Daemon
Inside the target Kali Linux machine, install and initialize the OpenSSH Server utility:

Bash
sudo apt update && sudo apt install openssh-server -y
sudo systemctl enable ssh --now
Verify the daemon is listening for traffic on port 22:

Bash
sudo systemctl status ssh
Step 2: Establish Virtual Networking (VirtualBox)
To ensure the host machine can route traffic into the background instance:

Open the virtual machine settings panel.

Attach a secondary network adapter set to Host-Only Adapter (e.g., vboxnet0).

Boot the VM and query the assigned private interface IP address by executing:

Bash
ip a
(Note down the target IP address, typically 192.168.56.X)

Step 3: Initialize the VS Code Remote Tunnel
Launch Visual Studio Code on your host workstation.

Navigate to the Extensions marketplace, find, and install Remote - SSH (developed by Microsoft).

Tap Ctrl+Shift+P (or Cmd+Shift+P on Mac) to call up the command palette and select:

Plaintext
Remote-SSH: Connect to Host...
Input your connection string payload: kali@192.168.56.X (replace with your active VM IP).

Provide your system account authentication password when prompted.

VS Code will automatically bootstrap a headless workspace background agent inside the VM. You can now use the integrated workspace console to trigger execution loops natively:

Bash
python main.py --input data/sample_broadcast.mp4
