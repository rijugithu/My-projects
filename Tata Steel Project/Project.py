from typing import List
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import pandas as pd
import shap
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from starlette.responses import StreamingResponse
from zipfile import ZipFile
import logging

# Initialize SHAP visualization
shap.initjs()
# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/upload_excel/")
async def upload_excel(
        file: UploadFile = File(...),
        target_columns: str = Form(...),
        feature_columns: str = Form(...)
):
    try:
        # Read the uploaded file
        content = await file.read()
        df = pd.read_excel(io.BytesIO(content))

        # Log the DataFrame columns for debugging
        logger.info(f"DataFrame columns: {df.columns.tolist()}")

        # Parse target and feature columns
        target_columns = [col.strip() for col in target_columns.split(',')]
        feature_columns = [col.strip() for col in feature_columns.split(',')]

        # Log the parsed target and feature columns for debugging
        logger.info(f"Parsed target columns: {target_columns}")
        logger.info(f"Parsed feature columns: {feature_columns}")

        # Validate target and feature columns
        if not set(target_columns).issubset(df.columns) or not set(feature_columns).issubset(df.columns):
            raise HTTPException(status_code=400, detail="Invalid target or feature columns")

        # Impute missing values
        imputer = KNNImputer(n_neighbors=3)
        df_imputed = imputer.fit_transform(df)
        df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

        y = df_imputed[target_columns]
        X = df_imputed[feature_columns]

        images = []
        for target_column in target_columns:
            y_target = y[target_column].values

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=0.3, random_state=42)

            # Scale data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train KNN regressor
            knn_regressor = KNeighborsRegressor(n_neighbors=5)
            knn_regressor.fit(X_train_scaled, y_train)
            y_pred = knn_regressor.predict(X_test_scaled)

            # Plot actual vs predicted
            plt.figure(figsize=(10, 6))
            plt.plot(y_test, label='Y-actual', marker='o')
            plt.plot(y_pred, label='Y-pred', marker='x')
            plt.xlabel('Actual Y')
            plt.ylabel('Predicted Y')
            plt.title(f'Line Plot of Actual Y vs Predicted Y for {target_column}')
            plt.legend()
            plt.grid(True)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            images.append((buf, f'{target_column}_knn.png'))
            plt.close()

            # Train XGBoost regressor
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)

            # Perform SHAP analysis
            explainer = shap.Explainer(model)
            shap_values = explainer(X_train_scaled)
            shap.waterfall_plot(shap_values[0])

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            images.append((buf, f'{target_column}_shap.png'))
            plt.close()

        # Create ZIP file of images
        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, 'w') as zf:
            for img_buf, img_name in images:
                zf.writestr(img_name, img_buf.getvalue())
        zip_buffer.seek(0)

        return StreamingResponse(zip_buffer, media_type="application/x-zip-compressed",
                                 headers={"Content-Disposition": "attachment; filename=images.zip"})

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
