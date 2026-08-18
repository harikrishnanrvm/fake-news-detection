# User Manual

**Phase:** Phase 7 — Deployment and User Interface

This document explains how to use the Fake News Detection web page. It assumes the application
is already running - see `docs/deployment_guide.md` if it isn't started yet.

---

## 1. How to Start the Application

1. Open a terminal in the project folder and activate the virtual environment.
2. Run `uvicorn api.app:app --reload`.
3. Wait until the terminal shows `Application startup complete.`
4. Open `http://127.0.0.1:8000/app` in a web browser.

## 2. How to Paste an Article

The page has one large text box under "Paste News Article". Click inside it and paste (or type)
the full text of a news article.

A few points to keep in mind:

- The article needs to be **at least 50 characters** long. A single short sentence or a headline
  by itself will usually be rejected as too short.
- The article can be **up to 20,000 characters**. This is more than enough for almost any news
  article - it's roughly 3,000-4,000 words.
- Two ready-made sample articles are available in the `demo/` folder
  (`fake_article.txt` and `real_article.txt`) if you want something to try immediately without
  finding your own article.

## 3. How to Predict

Click the **Predict** button below the text box.

While the prediction is running, the button becomes disabled and shows **Predicting...** - this
usually takes well under a second. Once it's done, the button goes back to normal and a new
section appears below showing:

- **Prediction** - either `REAL` or `FAKE`
- **Confidence** - how sure the model is about that prediction, as a percentage
- **Model** - which model produced the result (currently `LSTM`)

To try a different article, either select all the text in the box and replace it, or click
**Clear** to empty the box and hide the previous result.

## 4. Understanding the Confidence Score

The confidence score tells you how strongly the model leans toward its answer - it is **not** a
measure of how factually correct the article is.

For example, a result of `REAL` at `99.73%` confidence means the model is very sure this article
looks like the kind of writing it learned to associate with "Real" during training - not that
every fact in the article has been checked and confirmed true.

A confidence score close to 50% (for example, 55% or 60%) means the model is unsure - the article
doesn't clearly match either pattern it learned. In these cases, treat the prediction as a weak
signal rather than a confident answer.

## 5. Current Limitations

It's important to know what this tool can and cannot do, so results aren't misread:

- **It does not fact-check anything.** The model was trained to recognize writing patterns
  associated with the "Fake" and "Real" articles in its training dataset, not to verify specific
  claims against real-world facts.
- **It only works reliably on English text.** The model was trained only on English news
  articles. Pasting text in another language will still produce a prediction (the app does not
  reject it), but that prediction is not meaningful.
- **It works best on news-style writing.** The model was trained on political news articles from
  around 2016-2017. Very different kinds of text (social media posts, opinion blogs on unrelated
  topics, academic papers) may not be classified reliably, since they don't resemble what the
  model was trained on.
- **A high confidence score does not guarantee a correct answer.** Like any Machine Learning
  model, this one makes mistakes - see `docs/model_comparison_report.md` for the actual error
  rates measured during testing.
- **This is a student project**, not a production fact-checking service. It should not be used as
  the sole basis for deciding whether a real news article is true or false.

## 6. If Something Goes Wrong

The page shows a plain, friendly message instead of a technical error whenever something goes
wrong - for example:

- **"Please enter a news article."** - the text box was empty when Predict was clicked.
- **"Article text must be at least 50 characters long."** - the pasted text was too short.
- **"Unable to contact server."** - the browser could not reach the backend at all. Check that
  the `uvicorn` server is still running in its terminal window.

If you see any of these, they are expected behaviour, not bugs - just follow the message's
instructions and try again.
