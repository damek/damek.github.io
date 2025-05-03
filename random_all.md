---
layout: default
title: "random notes – view all"
permalink: /random/all/
---

## random notes — full feed  

{% assign notes = site.random | sort: "date" | sort: "updated" | reverse %}
{% for n in notes %}
---
### {{ n.title }}
<time>{{ n.updated | default:n.date | date: "%Y-%m-%d" }}</time>

{::nomarkdown}
{{ n.content }}
{:/nomarkdown}

{% endfor %}