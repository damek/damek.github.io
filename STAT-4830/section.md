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
  const hash = window.location.hash; // Get the hash from URL
  
  if (lectureNum !== null) {
    try {
      // Handle both old and new formats
      let url;
      if (lectureNum.includes('.')) {
        // New format with sections (e.g., 1.1)
        url = `https://raw.githubusercontent.com/damek/STAT-4830/main/section/${lectureNum}/notes.md`;
      } else {
        // Try new format first (e.g., section/0/notes.md)
        url = `https://raw.githubusercontent.com/damek/STAT-4830/main/section/${lectureNum}/notes.md`;
      }
      
      console.log('Fetching from:', url);
      let response = await fetch(url);
      
      // If new format fails and it's a number without decimal, try old format
      if (!response.ok && !lectureNum.includes('.')) {
        url = `https://raw.githubusercontent.com/damek/STAT-4830/main/Lecture${lectureNum}.md`;
        console.log('Trying old format:', url);
        response = await fetch(url);
      }
      
      console.log('Response status:', response.status);
      
      if (response.ok) {
        let content = await response.text();
        
        // Pre-process image URLs - now handles section-based figures
        content = content.replace(
          /!\[(.*?)\]\((.*?)\)/g,
          (match, alt, src) => {
            if (src && !src.startsWith('http')) {
              const sectionPath = lectureNum.includes('.') ? 
                `section/${lectureNum}/` : 
                `section/${lectureNum}/`;
              
              // If it's a relative path starting with ./ or ../, resolve it relative to the current section
              if (src.startsWith('./') || src.startsWith('../')) {
                const resolvedPath = new URL(src, `https://raw.githubusercontent.com/damek/STAT-4830/main/${sectionPath}`).pathname.slice(1);
                return `![${alt}](https://raw.githubusercontent.com/damek/STAT-4830/main/${resolvedPath})`;
              }
              
              // For other paths, try section directory first, then fall back to repo root
              if (!src.startsWith('/')) {
                return `![${alt}](https://raw.githubusercontent.com/damek/STAT-4830/main/${sectionPath}${src})`;
              }
              
              // For absolute paths (starting with /), use from repo root
              return `![${alt}](https://raw.githubusercontent.com/damek/STAT-4830/main/${src.slice(1)})`;
            }
            return match;
          }
        );
        
        document.getElementById('lecture-content').innerHTML = content;
        await processGitHubContent(document.getElementById('lecture-content'));
        
        // Initialize syntax highlighting
        document.querySelectorAll('pre code').forEach((block) => {
          hljs.highlightBlock(block);
        });

        // Process math
        if (window.MathJax) {
          await MathJax.typesetPromise();
        }

        // After everything is loaded and processed, scroll to hash if present
        if (hash) {
          const element = document.querySelector(hash);
          if (element) {
            // Add a small delay to ensure everything is rendered
            setTimeout(() => {
              element.scrollIntoView();
            }, 100);
          }
        }
      } else {
        console.error('Response not OK:', await response.text());
        document.getElementById('lecture-content').innerHTML = 
          `<p>Notes for section ${lectureNum} not found. Please check that the file exists in the repository.</p>`;
      }
    } catch (error) {
      console.error('Detailed error:', error);
      document.getElementById('lecture-content').innerHTML = 
        `<p>Error loading notes content: ${error.message}</p>`;
    }
  } else {
    document.getElementById('lecture-content').innerHTML = 
      '<p>No section number specified. Please use ?n=X in the URL.</p>';
  }
}

document.addEventListener('DOMContentLoaded', fetchLecture);
</script> 