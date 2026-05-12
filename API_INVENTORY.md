# ProgramAT API Setup Manual

## What to edit

This project uses three API credentials: `GEMINI_API_KEY` (required for AI parsing/vision features), Google Cloud Vision credentials `GOOGLE_APPLICATION_CREDENTIALS` (Live OCR with Cloud Vision), and GitHub integration `GITHUB_TOKEN`.

### Edit .env file

Open [backend/.env](backend/.env) (delete '.example' in the name) and set only these values:

```env
GITHUB_TOKEN=your_github_token_here

GEMINI_API_KEY=your_gemini_api_key_here

GOOGLE_APPLICATION_CREDENTIALS=/full/path/to/your-google-cloud-vision-credentials.json
```

These values are left unchanged:

```env
GITHUB_REPO=program-at/ProgramAT-opensource

GEMINI_MODEL=gemini-3-flash-preview

HOST=0.0.0.0
PORT=8080
PAUSE_DURATION=5.0
```

### Set as environment variables in shell

If you do not want to use `backend/.env`, you can set the same values as environment variables in the shell before starting the backend.

Linux / macOS:

```bash
export GITHUB_TOKEN="your_github_token_here"
export GEMINI_API_KEY="your_gemini_api_key_here"
export GOOGLE_APPLICATION_CREDENTIALS="/full/path/to/your-google-cloud-vision-credentials.json"
cd backend
python stream_server.py
```

Windows PowerShell:

```powershell
$env:GITHUB_TOKEN = "your_github_token_here"
$env:GEMINI_API_KEY = "your_gemini_api_key_here"
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\credentials.json"
cd backend
python stream_server.py
```

## How to find the keys

### GitHub Token

1. Go to any GitHub page: <https://github.com/>. Click your profile picture (top right) and open **Settings**.
2. Go to **Developer settings** -> **Personal access tokens** -> **Tokens (classic)**.
3. Click **Generate new token**.
4. Add a note, choose an expiration, and select scopes to be **repo**.
5. Click **Generate token**, copy it and save it properly. You won't be able to see it again.
6. Paste the token into `backend/.env` as `GITHUB_TOKEN`.

### Gemini API Key

1. Go to Google AI Studio: <https://aistudio.google.com>
2. Sign in with your Google account.
3. In the left sidebar, click **Get API key** -> **Create API key**.
4. Select an existing Google Cloud project or create a new one, then click **Create API key**.
5. Copy the generated key.
6. Paste the key into `backend/.env` as `GEMINI_API_KEY`.

Billing:

- If you want to increase the rate limit, click **Set up billing** in Google AI Studio and choose other plans for your API key.

### Google Application Credentials

This project uses Google Cloud Vision for higher-quality OCR in `tools/live_ocr.py`. If you want Cloud Vision for better OCR, create a service account and download a JSON key, then set `GOOGLE_APPLICATION_CREDENTIALS` to that file path. If you prefer not to use Cloud Vision, the tool will fall back to local Tesseract if installed (no Google credentials required).

Steps:

1. Go to Google Cloud Console: <https://console.cloud.google.com>
2. Create or select a project and enable the **Cloud Vision API** in **Menu** -> **API & Services** -> **Enabled APIs & services**.
3. Create a service account: **Menu** -> **IAM & Admin** -> **Service Accounts** -> **Create service account**. Fill name/ID and click **Continue**.
4. Create a key: for this service account, select **Actions** -> **Manage Keys** -> **Add key** -> **Create new key** -> choose **JSON** -> **Create**. A JSON file `credentials.json` will be downloaded. Save it securely into your workspace.
5. Set the environment variable to point to the JSON file.

```env
GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\credentials.json
```

Security notes:

- Do not commit the JSON key to version control. Add `credentials.json` to `.gitignore`.
- Restrict the service account to only the Vision API and grant the minimal needed permissions.
