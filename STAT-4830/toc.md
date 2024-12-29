---
layout: course_page
title: "STAT 4830: Table of Contents"
---

{% assign repo_owner = "damek" %}
{% assign repo_name = "STAT-4830" %}

{% assign toc_url = "https://api.github.com/repos/" | append: repo_owner | append: "/" | append: repo_name | append: "/contents/toc.md" %}

<div class="toc-content">
  <!-- Content will be dynamically loaded from GitHub -->
</div>

<script>
async function fetchTocContent() {
  try {
    const response = await fetch('https://api.github.com/repos/damek/STAT-4830/contents/toc.md');
    const data = await response.json();
    
    if (data.content) {
      const content = atob(data.content); // Decode base64 content
      const tocContent = document.querySelector('.toc-content');
      tocContent.innerHTML = marked.parse(content);
    } else {
      document.querySelector('.toc-content').innerHTML = `
        <h2>Course Contents</h2>
        <p><em>The table of contents will be available when the course materials are published.</em></p>
      `;
    }
  } catch (error) {
    console.error('Error fetching TOC:', error);
    document.querySelector('.toc-content').innerHTML = `
      <h2>Course Contents</h2>
      <p><em>The table of contents will be available when the course materials are published.</em></p>
    `;
  }
}

document.addEventListener('DOMContentLoaded', fetchTocContent);
</script> 