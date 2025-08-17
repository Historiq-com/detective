


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
