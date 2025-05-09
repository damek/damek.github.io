---
layout: default
title: "Notes"
permalink: /random/
---

# Notes


This is an attempt to collate notes I write while I'm researching or scrolling online. Expect small summaries of papers / technical results and half-baked thoughts. Maybe it will be useful to you? You shouldn't take them too seriously. 

{% assign updated_notes = site.random
     | where_exp:"d","d.updated"
     | sort:"updated" | reverse %}

{% assign fresh_notes = site.random
     | where:"updated", nil
     | sort:"date" | reverse %}

{% assign notes = updated_notes | concat: fresh_notes %}

<ul class="notes">
{% for n in notes %}
<li class="note-item">
  <div class="note-row">
    <span class="note-date">{{ n.updated | default:n.date | date:"%Y-%m-%d" }}</span>
    <a class="note-title" href="{{ n.url }}">{{ n.title }}</a>
  </div>

  {% if n.description %}
    <blockquote class="note-desc">{{ n.description }}</blockquote>
  {% endif %}

  {% if n.tags %}
    <div class="chips">
      {% for t in n.tags %}
        <a href="/random/tags/#{{ t | slugify }}">{{ t }}</a>
      {% endfor %}
    </div>
  {% endif %}
</li>


{% endfor %}
</ul>
