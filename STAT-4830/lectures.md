---
layout: course_page
---
<style>
/* Override list spacing for lecture content only */
#lecture-content ul,
#lecture-content ol {
  margin: 0;
  padding-left: 1.5em;
  list-style-position: outside;
}

#lecture-content li {
  margin: 0;
  line-height: 1.4;
  padding-left: 0.2em;
}

#lecture-content li p {
  margin: 0;
}

#lecture-content p {
  margin: 0.8em 0;
}

/* Handle indented lists */
#lecture-content ul ul,
#lecture-content ul ol,
#lecture-content ol ul,
#lecture-content ol ol {
  margin: 0;
  padding-left: 1.2em;
}

/* Ensure proper alignment of list markers */
#lecture-content ul {
  list-style: disc outside none;
}

#lecture-content ul ul {
  list-style-type: circle;
}

#lecture-content ul ul ul {
  list-style-type: square;
}

/* Add minimal spacing between sections */
#lecture-content h1,
#lecture-content h2,
#lecture-content h3,
#lecture-content h4 {
  margin-top: 1em;
  margin-bottom: 0.5em;
}
</style>

<div id="lecture-content">
  Loading lecture content...
</div>

<script>
async function fetchLecture() {
  const urlParams = new URLSearchParams(window.location.search);
  const lectureNum = urlParams.get('n');
  
  if (lectureNum !== null) {
    try {
      const url = `https://raw.githubusercontent.com/damek/STAT-4830/main/Lecture${lectureNum}.md`;
      console.log('Fetching from:', url);
      
      const response = await fetch(url);
      console.log('Response status:', response.status);
      
      if (response.ok) {
        let content = await response.text();
        
        // Pre-process image URLs
        content = content.replace(
          /!\[(.*?)\]\((.*?)\)/g,
          (match, alt, src) => {
            if (src && !src.startsWith('http')) {
              return `![${alt}](https://raw.githubusercontent.com/damek/STAT-4830/main/${src})`;
            }
            return match;
          }
        );
        
        // Parse the content
        const parsedContent = marked.parse(content);
        document.getElementById('lecture-content').innerHTML = parsedContent;
        
        // Add IDs to headings after content is rendered
        document.querySelectorAll('#lecture-content h1, #lecture-content h2, #lecture-content h3, #lecture-content h4').forEach(heading => {
          const id = heading.textContent
            .toLowerCase()
            .replace(/[^\w\s-]/g, '')
            .replace(/\s+/g, '-');
          heading.id = id;
        });
        
        // Initialize syntax highlighting
        document.querySelectorAll('pre code').forEach((block) => {
          hljs.highlightBlock(block);
        });

        // Process math
        if (window.MathJax) {
          MathJax.typesetPromise();
        }
      } else {
        console.error('Response not OK:', await response.text());
        document.getElementById('lecture-content').innerHTML = 
          `<p>Lecture ${lectureNum} not found. Please check that the lecture file exists in the repository.</p>`;
      }
    } catch (error) {
      console.error('Detailed error:', error);
      document.getElementById('lecture-content').innerHTML = 
        `<p>Error loading lecture content: ${error.message}</p>`;
    }
  } else {
    document.getElementById('lecture-content').innerHTML = 
      '<p>No lecture number specified. Please use ?n=X in the URL.</p>';
  }
}

document.addEventListener('DOMContentLoaded', fetchLecture);
</script> 