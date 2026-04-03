---
layout: oldsite_blue_preview
title: Damek Davis
---
{% assign old_homepage = site.pages | where: "name", "index_old.md" | first %}
{{ old_homepage.content | markdownify }}
