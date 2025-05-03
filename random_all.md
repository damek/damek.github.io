---
layout: default
title: "random notes – view all"
permalink: /random/all/
---

## random notes — full feed  

{% assign notes = site.random | sort: "date" | reverse %}
{% for n in notes %}
---

### {{ n.title | default: n.slug }}  
<time datetime="{{ n.date }}">{{ n.date | date: "%Y-%m-%d" }}</time>

{::nomarkdown}
{{ n.content }}
{:/nomarkdown}


{% endfor %}