---
layout: default
title: Tags
permalink: /random/tags/
---

<h2>All tags</h2>
{% assign all_tags = site.random | map: "tags" | join: "," | split: "," | sort %}
{% assign uniq = "" | split: "" %}
{% for t in all_tags %}
  {% unless uniq contains t %}
    {% assign uniq = uniq | push: t %}
  {% endunless %}
{% endfor %}

<ul class="tag-list">
{% for tag in uniq %}
  <li>
    <a href="#{{ tag | slugify }}">{{ tag }}</a>
  </li>
{% endfor %}
</ul>

<hr>

{% for tag in uniq %}
<h3 id="{{ tag | slugify }}">{{ tag }}</h3>
<ul class="notes">
  {% for n in site.random %}
    {% if n.tags contains tag %}
      <li><a href="{{ n.url }}">{{ n.title }}</a>
          <span class="note-date">{{ n.updated | default:n.date | date: "%Y-%m-%d" }}</span>
      </li>
    {% endif %}
  {% endfor %}
</ul>
{% endfor %}
