# Screenshots Needed for the Dissertation

I could not capture these myself (no browser access from this environment), so they need to be
taken manually before submitting the report. The server must be running first — see
`docs/deployment_guide.md`.

Save each one in this folder using the file name given below, so they are easy to find later.

1. **`home_page.png`** — Open `http://127.0.0.1:8000/static/index.html` (or `/app`) in a browser,
   before typing anything. Shows the empty text box and the Predict/Clear buttons.

2. **`successful_prediction.png`** — Paste the contents of `demo/real_article.txt` (or
   `demo/fake_article.txt`) into the text box, click Predict, and wait for the result to appear.
   Shows the Prediction / Confidence / Model section filled in.

3. **`validation_error.png`** — Click Predict with the text box empty (or with only a few words
   typed in). Shows the friendly red error message.

4. **`swagger_docs.png`** — Open `http://127.0.0.1:8000/docs`. Shows the three endpoints
   (`/`, `/health`, `/predict`) listed in Swagger UI.

5. **`health_endpoint.png`** — Open `http://127.0.0.1:8000/health` directly in the browser. Shows
   the plain `{"status": "ok"}` JSON response.
