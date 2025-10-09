# Easy GitHub Upload Guide

Since Git command line isn't working yet, here are easier alternatives:

## Option 1: GitHub Desktop (Easiest)

1. **Download GitHub Desktop**: https://desktop.github.com/
2. **Install and sign in** with your GitHub account
3. **Click "Add an Existing Repository"**
4. **Browse to your folder**: `C:\Users\Franc\OneDrive - Universidad Adolfo Ibanez\Desktop\TIDlll`
5. **Click "create a repository"** if prompted
6. **Name your repository**: `celegans-neural-analysis`
7. **Add a commit message**: "Initial commit: C. elegans neural analysis"
8. **Click "Publish repository"**
9. **Make sure "Keep this code private" is unchecked** if you want it public

## Option 2: Web Upload (Manual but works)

1. **Go to GitHub.com** and create a new repository
2. **Name it**: `celegans-neural-analysis`
3. **Don't initialize with README** (we have one)
4. **Click "uploading an existing file"**
5. **Drag these files from your folder**:
   - `README.md`
   - `requirements.txt`
   - `.gitignore`
   - `QUICKSTART.md`
   - `PCA.py`
   - `Raster_Plot.py`
   - `Tuning_Curves.py`
   - `GCaMP_vs_Time.py`
   - `Analisis NWB File.ipynb`
   - `PCA_Analysis_Complete.png`

## Option 3: Try Git Again (After Terminal Restart)

1. **Close this PowerShell window completely**
2. **Open a new PowerShell window**
3. **Navigate back to your folder**:
   ```powershell
   cd "C:\Users\Franc\OneDrive - Universidad Adolfo Ibanez\Desktop\TIDlll"
   ```
4. **Try**: `git --version`
5. **If it works, run**: `.\setup_git.bat`

## Recommended: Use GitHub Desktop

GitHub Desktop is the easiest option - it provides a visual interface and handles all the Git commands for you automatically.

Once uploaded, your repository will be available at:
`https://github.com/yourusername/celegans-neural-analysis`

## Files NOT to upload:
- `venv/` folder (virtual environment)
- `.nwb` files (too large for GitHub)
- `.idea/` folder (IDE settings)

These are already excluded in the `.gitignore` file.