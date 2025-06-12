---
layout: default
title: "Notes"
permalink: /random/
---

# Notes


This is an attempt to collate notes I write while I'm researching or scrolling online. Expect small summaries of papers / technical results and half-baked thoughts. In that way it will be similar to my [grad school blog](https://damekdavis.wordpress.com/), in which I [wrote](https://damekdavis.wordpress.com/2011/04/16/a-note-on-sheafification/){:target="_blank"} [about](https://damekdavis.wordpress.com/2011/04/07/a-slightly-different-proof-of-burnsides-theorem/){:target="_blank"} [a bunch](https://damekdavis.wordpress.com/2011/03/25/parkers-lemma-is-equivalent-to-burnsides-lemma/){:target="_blank"} of [random](https://damekdavis.wordpress.com/2011/01/18/a-probabilistic-approach-to-group-commutativity/){:target="_blank"} [topics](https://damekdavis.wordpress.com/2010/12/13/is-the-product-of-measurable-spaces-the-categorical-product/){:target="_blank"}. Maybe it will be useful to you? You shouldn't take them too seriously.

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
