---
layout: course_page
title: "STAT 4830: Table of Contents"
---

{% include process_content.html %}

<div id="toc-content">
  Loading table of contents...
</div>

<script>
async function fetchTOC() {
  try {
    const response = await fetch('https://raw.githubusercontent.com/damek/STAT-4830/main/toc.md');
    const content = await response.text();
    
    document.getElementById('toc-content').innerHTML = content;
    await processGitHubContent(document.getElementById('toc-content'));
  } catch (error) {
    console.error('Error fetching TOC:', error);
    document.getElementById('toc-content').innerHTML = 
      '<p>Error loading table of contents. Please try again later.</p>';
  }
}

document.addEventListener('DOMContentLoaded', fetchTOC);
</script> 