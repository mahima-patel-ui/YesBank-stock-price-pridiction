# Yes Bank Stock Price Prediction — Streamlit

## Project
Regression model to predict Yes Bank closing price using:
Open, High, Low, Month and Year.

The final model in the supplied notebook is Linear Regression.

## Files
- `app.py` — Streamlit application
- `requirements.txt` — deployment dependencies
- `data_YesBank_StockPrices.csv` — place the original project dataset here

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app accepts the original CSV through the sidebar uploader.

## GitHub

```bash
git init
git add .
git commit -m "Add Yes Bank stock prediction Streamlit app"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

## Streamlit Community Cloud

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Select the GitHub repository.
4. Choose `app.py` as the main file.
5. Deploy.

Upload the original dataset in the app sidebar if it is not committed to GitHub.

## Important project limitation
The model predicts closing price from Open, High, Low, Month and Year. For a genuinely future date, Open/High/Low are not known in advance, so the app should be described as a price-estimation/demo application rather than guaranteed future stock forecasting.
