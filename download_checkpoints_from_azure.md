# Download Checkpoint Weights From Azure VM

## Overview
Use one of these methods to copy trained checkpoint weights from your Azure VM to your local machine:
1. Direct `scp` (single file)
2. Folder `scp` (all checkpoints)
3. Compress on VM, then download
4. Azure Blob Storage (for bigger runs)

## 1) Direct scp (single checkpoint)
From your local machine (PowerShell):

```powershell
scp azureuser@<VM_PUBLIC_IP>:/home/azureuser/bazaar/checkpoints/best.pt .\best.pt
```

With SSH key:

```powershell
scp -i C:\path\to\id_rsa azureuser@<VM_PUBLIC_IP>:/home/azureuser/bazaar/checkpoints/best.pt .\best.pt
```

## 2) Download entire checkpoints directory
From your local machine:

```powershell
scp -r azureuser@<VM_PUBLIC_IP>:/home/azureuser/bazaar/checkpoints .\
```

## 3) Compress on VM, then download
On the Azure VM:

```bash
cd /home/azureuser/bazaar
tar -czf checkpoints.tar.gz checkpoints
```

On local machine:

```powershell
scp azureuser@<VM_PUBLIC_IP>:/home/azureuser/bazaar/checkpoints.tar.gz .\
```

Then extract locally.

## 4) Azure Blob route (optional)
Good when you have many checkpoints or unstable SSH sessions.

1. Upload checkpoint artifacts from VM to an Azure Storage container.
2. Download locally via SAS URL.

## Verify on VM before download
Check files exist:

```bash
ls -lah /home/azureuser/bazaar/checkpoints
```

Check newest files:

```bash
ls -lt /home/azureuser/bazaar/checkpoints | head
```

Optional integrity hash:

```bash
sha256sum /home/azureuser/bazaar/checkpoints/best.pt
```

## Use downloaded checkpoint locally
Place `best.pt` under local `checkpoints/` and run evaluation:

```powershell
python -m src.evaluate --checkpoint checkpoints/best.pt --skip-isolation --calibrate-quantiles
```
