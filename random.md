---
layout: default
title: "Notes"
permalink: /random/
---

# Notes


This is an attempt to collate notes I write while I'm doing research or scrolling online. Expect small summaries of papers / technical results and half-baked thoughts. Maybe it will be useful to someone else? You shouldn't take them too seriously. You should also know that I may have collaborated with a language model to help me organize my thoughts and decide what not to write.

{% assign notes = site.random | sort:"date" | sort:"updated" | reverse %}

<ul class="notes">
{% for n in notes %}
  <li class="note-item">
    <a class="note-title" href="{{ n.url }}">{{ n.title }}</a>
    <span class="note-date">{{ n.updated | default:n.date | date: "%Y-%m-%d" }}</span>

    {% if n.tags %}
      <span class="chips">
        {% for t in n.tags %}
          <a href="/random/tags/#{{ t | slugify }}">{{ t }}</a>{% unless forloop.last %},{% endunless %}
        {% endfor %}
      </span>
    {% endif %}

    {% if n.description %}
      <blockquote class="note-desc">{{ n.description }}</blockquote>
    {% endif %}
  </li>
{% endfor %}
</ul>
