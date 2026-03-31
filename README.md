# 🚀 AI Career Optimization OpenENV

## 📌 Overview

This project is a real-world simulation environment built using OpenEnv, where an AI agent learns how to optimize a candidate's career path (skills, projects, experience) to maximize job success probability.

---

## 🎯 Problem Statement

Choosing the right skills and actions to land a job is complex.
This environment simulates real-world decision-making for career growth.

---

## ⚙️ Features

* 🔄 **Step-based environment** (`/step`)
* 📊 **State tracking** (`/state`)
* 🧠 **Automatic grading system** (`/auto_grader`)
* 🎯 **Job matching logic**
* 📈 **Reward-based learning system**
* 🔁 **Resettable simulation**

---

## 🧪 API Endpoints

* `POST /reset` → Start new episode
* `POST /step` → Perform action (learn skill, apply job, etc.)
* `GET /state` → Get current state
* `POST /grader` → Stateless evaluation
* `GET /auto_grader` → Automatic evaluation

---

## 🤖 Inference

Run:

```bash
python inference.py
```

This will:

* Simulate agent actions
* Evaluate performance using grader

---

## 🏗️ Tech Stack

* Python
* FastAPI
* OpenEnv
* Hugging Face Spaces (Docker)

---

## 🌐 Live Demo

👉 https://yeswanth29-aicareeroptimizationopenenv.hf.space

---

## 📂 GitHub Repository

👉 https://github.com/yeswanthvuddagiri/AICareerOptimizationOpenENV

---

## 💡 Key Highlight

Supports both:

* Stateless grading (`/grader`)
* Automatic grading (`/auto_grader`)

Ensuring reliability in distributed environments like Hugging Face Spaces.

---

## 🏆 Conclusion

This project demonstrates how AI can simulate and optimize real-world career decisions using reinforcement learning concepts.
