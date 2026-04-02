---
layout: hadamard_preview
title: Damek Davis
---
{% assign homepage = site.pages | where: "url", "/" | first %}
{{ homepage.content | markdownify }}
