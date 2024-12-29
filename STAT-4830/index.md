---
layout: course_page
---

<div id="index-content">
  Loading course content...
</div>

<script>
async function fetchReadme() {
  try {
    const response = await fetch('https://api.github.com/repos/damek/STAT-4830/readme');
    const data = await response.json();
    const content = atob(data.content);
    
    document.getElementById('index-content').innerHTML = content;
    await processGitHubContent(document.getElementById('index-content'));
  } catch (error) {
    console.error('Error fetching README:', error);
    document.getElementById('index-content').innerHTML = 
      '<p>Error loading course content. Please try again later.</p>';
  }
}

document.addEventListener('DOMContentLoaded', fetchReadme);
</script> 