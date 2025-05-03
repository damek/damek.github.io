---
layout: none           # no wrapper; we just redirect
permalink: /random/pick/
---


<script>
/* list of all random-note URLs, baked in by Liquid */
const notes = [
  {% for n in site.random %}
    "{{ n.url | url_encode }}"{% unless forloop.last %},{% endunless %}
  {% endfor %}
];

/* choose & jump */
const target = notes[Math.floor(Math.random() * notes.length)];
window.location.href = decodeURIComponent(target);
</script>

<noscript>
  JavaScript required.  
  <ul>
    {% for n in site.random %}
      <li><a href="{{ n.url }}">{{ n.title | default:n.slug }}</a></li>
    {% endfor %}
  </ul>
</noscript>
