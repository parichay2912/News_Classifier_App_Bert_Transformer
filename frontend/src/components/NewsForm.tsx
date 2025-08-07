import { useState } from "react";
import { predictArticle, generateSummary, improveHeadline } from "../api";
import "./NewForm.css";

const NewsForm = () => {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");

  const [category, setCategory] = useState("");
  const [summary, setSummary] = useState("");
  const [improvedTitle, setImprovedTitle] = useState("");
  const [confidence, setconfidence] = useState("");

  const [modelName, setModelName] = useState("bert-base-uncased");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const result = await predictArticle({
      title,
      description,
      model_name: modelName,
    });
    setCategory(result.predicted_class_label || result.category || "");
    setconfidence(result.confidence|| "")
  };

  const handleSummary = async () => {
    const result = await generateSummary({ title, description });
    setSummary(result.summary);
  };

  const handleImprovedTitle = async () => {
    const result = await improveHeadline({ title, description });
    setImprovedTitle(result.improved_headline || result.improved_title || "");
  };

  return (
    <div className="news-form-container">
      <h2>News Analyzer</h2>

      <form onSubmit={handleSubmit}>
        <input
          placeholder="Title"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          required
        />

        <textarea
          placeholder="Description"
          rows={5}
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          required
        />
        

        <select
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
        >
          <option value="bert-base-uncased">bert-base-uncased</option>
          <option value="distilbert-base-uncased">distilbert-base-uncased</option>
          <option value="roberta-base">roberta-base</option>
          <option value="albert-base-v2">albert-base-v2</option>
        </select>

        <button type="submit">Classify</button>
      </form>

      <button onClick={handleSummary} disabled={!title || !description}>
        Generate Summary
      </button>

      <button onClick={handleImprovedTitle} disabled={!title || !description}>
        Improve Headline
      </button>
      {confidence && <p><strong>Confidence:</strong> {confidence}</p>}
      {category && <p><strong>Category:</strong> {category}</p>}
      {summary && <p><strong>Summary:</strong> {summary}</p>}
      {improvedTitle && <p><strong>Improved Title:</strong> {improvedTitle}</p>}
    </div>
  );
};

export default NewsForm;
