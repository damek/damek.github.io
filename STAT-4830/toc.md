---
layout: course_page
title: "STAT 4830: Table of Contents"
---

<style>
/* Override list spacing for section content */
#section-content ul,
#section-content ol {
  margin: 0;
  padding-left: 1.5em;
  list-style-position: outside;
}

#section-content li {
  margin: 0;
  line-height: 1.4;
  padding-left: 0.2em;
}

#section-content li p {
  margin: 0;
}

#section-content p {
  margin: 0.8em 0;
}

/* Handle indented lists */
#section-content ul ul,
#section-content ul ol,
#section-content ol ul,
#section-content ol ol {
  margin: 0;
  padding-left: 1.2em;
}

/* Ensure proper alignment of list markers */
#section-content ul {
  list-style: disc outside none;
}

#section-content ul ul {
  list-style-type: circle;
}

#section-content ul ul ul {
  list-style-type: square;
}

/* Add minimal spacing between sections */
#section-content h1,
#section-content h2,
#section-content h3,
#section-content h4 {
  margin-top: 1em;
  margin-bottom: 0.5em;
}
</style>

<div id="section-content">
  Loading table of contents...
</div>

<script>
async function fetchTOC() {
  try {
    const response = await fetch('https://raw.githubusercontent.com/damek/STAT-4830/main/toc.md');
    const content = await response.text();
    
    document.getElementById('section-content').innerHTML = content;
    await processGitHubContent(document.getElementById('section-content'));
  } catch (error) {
    console.error('Error fetching TOC:', error);
    document.getElementById('section-content').innerHTML = 
      '<p>Error loading table of contents. Please try again later.</p>';
  }
}

document.addEventListener('DOMContentLoaded', fetchTOC);
</script> 