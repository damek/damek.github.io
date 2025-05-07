---
layout: default
title: "Notes – view all"
permalink: /random/all/
---

## Notes — full feed  

{% assign updated_notes = site.random
     | where_exp:"d","d.updated"
     | sort:"updated" | reverse %}

{% assign fresh_notes = site.random
     | where:"updated", nil
     | sort:"date" | reverse %}

{% assign notes = updated_notes | concat: fresh_notes %}{% for n in notes %}
---
### {{ n.title }}
<time>{{ n.updated | default:n.date | date: "%Y-%m-%d" }}</time>

{::nomarkdown}
{{ n.content }}
{:/nomarkdown}

{% endfor %}