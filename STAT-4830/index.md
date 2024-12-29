---
layout: course_page
title: "STAT 4830: Numerical Optimization for Data Science and Machine Learning"
---

{% assign repo_owner = "damek" %}
{% assign repo_name = "STAT-4830" %}

{% assign readme_url = "https://api.github.com/repos/" | append: repo_owner | append: "/" | append: repo_name | append: "/readme" %}

{% assign readme_content = readme_url | jsonify %}

<div class="course-content">
  {{ readme_content | markdownify }}
</div>

<script>
async function fetchReadme() {
  try {
    const response = await fetch('https://api.github.com/repos/damek/STAT-4830/readme');
    const data = await response.json();
    const content = atob(data.content); // Decode base64 content
    
    const courseContent = document.querySelector('.course-content');
    courseContent.innerHTML = marked.parse(content); // Parse markdown to HTML
  } catch (error) {
    console.error('Error fetching README:', error);
  }
}

// Load marked.js for markdown parsing
const script = document.createElement('script');
script.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
script.onload = fetchReadme;
document.head.appendChild(script);
</script> 