const BASE_URL = "http://localhost:8000";

export const getModelInfo = async () => {
  const response = await fetch(`${BASE_URL}/model-info`);
  return response.json();
};

export const predictArticle = async (data: {
  title: string;
  description: string;
  model_name?: string;
}) => {
  const response = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return response.json();
};

export const generateSummary = async (data: {
  title: string;
  description: string;
}) => {
  const response = await fetch(`${BASE_URL}/generate-summary`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return response.json();
};

export const improveHeadline = async (data: {
  title: string;
  description: string;
}) => {
  const response = await fetch(`${BASE_URL}/improve-headline`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return response.json();
};
