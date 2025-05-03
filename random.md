---
layout: default
title: "random notes"
permalink: /random/
---



# random notes

this is an attempt to collate some of the random notes i write while i'm doing research or scrolling online. many times they are small summaries of papers. other times they're a concept i hear about again and again and eventually decide to investigate. maybe it will be useful to someone else? you shouldn't take them too seriously. you should also know that i've likely also collaborated with a language model to help me write them in a succinct way. 


{% assign notes = site.random | sort: "date" | sort: "updated" | reverse %}
<ul>
{% for n in notes %}
  <li>
    <time>{{ n.updated | default:n.date | date: "%Y-%m-%d" }}</time>
    <a href="{{ n.url }}">{{ n.title }}</a>
  </li>
{% endfor %}
</ul>