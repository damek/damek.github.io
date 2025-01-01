require 'jekyll'
require 'net/http'
require 'json'
require 'base64'

module Jekyll
  class GitHubLectureGenerator < Generator
    safe true
    priority :high

    def generate(site)
      # GitHub API configuration for public repo
      owner = "damek"
      repo = "STAT-4830"
      api_url = "https://raw.githubusercontent.com/#{owner}/#{repo}/main"

      # Get list of files from the repository
      begin
        # First get the directory listing
        contents_url = "https://api.github.com/repos/#{owner}/#{repo}/contents"
        uri = URI(contents_url)
        response = Net::HTTP.get(uri)
        files = JSON.parse(response)

        # Get section directories
        section_dirs = files.select { |f| f['type'] == 'dir' && f['name'] == 'section' }
        
        if section_dirs.any?
          # Get contents of section directory
          section_url = "#{contents_url}/section"
          section_response = Net::HTTP.get(URI(section_url))
          section_contents = JSON.parse(section_response)
          
          # Process each section directory
          section_contents.each do |section|
            if section['type'] == 'dir'
              # Get notes.md from each section
              notes_url = "#{api_url}/section/#{section['name']}/notes.md"
              begin
                content_response = Net::HTTP.get(URI(notes_url))
                
                # Create a new page
                page = PageWithoutAFile.new(site, site.source, "STAT-4830", "section_#{section['name']}.md")
                page.content = content_response
                page.data['layout'] = 'course_page'
                page.data['title'] = "Section #{section['name']}"
                
                # Add the page to the site
                site.pages << page
              rescue => e
                Jekyll.logger.warn "GitHubLectureGenerator:", "Failed to fetch notes for section #{section['name']}: #{e.message}"
              end
            end
          end
        end

        # Also handle old lecture format for backward compatibility
        lecture_files = files.select { |f| f['name'] =~ /^Lecture\d+\.md$/ }
        lecture_files.each do |file|
          begin
            lecture_url = "#{api_url}/#{file['name']}"
            content_response = Net::HTTP.get(URI(lecture_url))
            
            # Create a new page
            page = PageWithoutAFile.new(site, site.source, "STAT-4830", file['name'])
            page.content = content_response
            page.data['layout'] = 'course_page'
            page.data['title'] = "Lecture #{file['name'].match(/\d+/)[0]}"
            
            # Add the page to the site
            site.pages << page
          rescue => e
            Jekyll.logger.warn "GitHubLectureGenerator:", "Failed to fetch #{file['name']}: #{e.message}"
          end
        end
      rescue => e
        Jekyll.logger.warn "GitHubLectureGenerator:", "Failed to fetch repository contents: #{e.message}"
      end
    end
  end
end 