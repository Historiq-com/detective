gcloud builds submit --tag gcr.io/$(gcloud config get-value project)/subjobj-sam2
gcloud run deploy subjobj-sam2 \
  --image gcr.io/$(gcloud config get-value project)/subjobj-sam2 \
  --region=us-central1 \
  --service-account=your-sa@vlgo-site-567f8.iam.gserviceaccount.com \
  --cpu=4 --memory=8Gi --min-instances=1 --timeout=1200 \
  --set-env-vars=GOOGLE_CLOUD_PROJECT=vlgo-site-567f8,GOOGLE_CLOUD_LOCATION=us-central1,ULTRA_SAM_WEIGHTS=sam2.1_b.pt \
  --allow-unauthenticated


docker run --rm -p 8080:8080 \
  -v $HOME/.config/gcloud:/root/.config/gcloud:ro \
  -e GOOGLE_CLOUD_PROJECT="vlgo-site-567f8" \
  -e GOOGLE_CLOUD_LOCATION="us-central1" \
  subjobj-sam2

  gcloud run deploy seg-api-dev \
  --source . \
  --region us-central1 \
  --service-account vertex-ai-app@vlgo-site-567f8.iam.gserviceaccount.com \
  --cpu 4 --memory 8Gi \
  --timeout 1200 \
  --no-allow-unauthenticated
